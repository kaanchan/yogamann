"""
sd_make.py -- worker (single task or work-list)
· SDXL + ControlNet-OpenPose-SDXL (DWPose-trained)
· PyTorch nightly cu130, bfloat16 (Blackwell native)
· Closes #7
"""

from __future__ import annotations
import argparse, json, logging, os, sys, time, warnings
from pathlib import Path
from typing import Dict, List

# Suppress noisy 3rd-party deprecation warnings before any imports fire them
warnings.filterwarnings("ignore", message=r"Overwriting tiny_vit_",          category=UserWarning)
warnings.filterwarnings("ignore", message=r"The module 'mediapipe'",          category=UserWarning)
warnings.filterwarnings("ignore", message=r"Specified provider 'CUDAExecutionProvider'", category=UserWarning)
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Please import.*scipy\.ndimage\.filters", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Importing from timm\.models\.(layers|registry)", category=FutureWarning)

import torch

_log_level = os.environ.get("YOGAMANN_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, _log_level),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
from PIL import Image, ImageOps, PngImagePlugin
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from dwpose_onnx import DWposeDetector

from extract_pose import make_controlnet_mask, save_keypoints
from create_contact_sheet import build_comparison_2, build_comparison_4
from db import open_db, sha256_file, ingest_json

BASE_ID    = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROL_ID = "xinsir/controlnet-openpose-sdxl-1.0"


def _vram() -> str:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        return f"{free/1e9:.1f}/{total/1e9:.1f} GB free"
    return "N/A"


def build_pipe(device: str):
    log.info("Loading ControlNet — %s (float16)", CONTROL_ID)
    cn = ControlNetModel.from_pretrained(CONTROL_ID, torch_dtype=torch.float16)

    log.info("Loading SDXL pipeline — %s (float16)", BASE_ID)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_ID,
        controlnet=cn,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    log.debug("VRAM after pipeline load: %s", _vram())

    try:
        import triton  # noqa: F401
        log.info("Compiling UNet with torch.compile (first run: ~2-5 min warmup)")
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
    except ImportError:
        log.warning("Triton not available — skipping torch.compile (install triton for ~2x speedup)")
    return pipe


def render(task: Dict, device: str, pipe) -> None:
    cfg   = task["cfg"]
    steps = int(cfg.get("steps") or 25)
    photo = Path(task["photo"])

    log.info("render — %s  steps=%d  guidance=%.1f  seed=%s",
             photo.name, steps, cfg["guidance"], cfg.get("seed", "random"))
    log.info("output  → %s", Path(task["output_png"]).resolve())

    # ------------------------------------------------------------------
    # 1. CONTROLNET MASK  (DWPose)
    # ------------------------------------------------------------------
    # Open once; exif_transpose corrects rotation stored in EXIF (common in phone photos)
    src_img    = ImageOps.exif_transpose(Image.open(task["photo"])).convert("RGB")
    source_sha = sha256_file(Path(task["photo"]))

    log.info("Detecting pose — %s", photo.name)
    t_pose = time.time()
    detector = DWposeDetector()
    control_img = detector(src_img)
    control_img.save(task["mask_png"])
    pose_s = time.time() - t_pose
    log.debug("Pose detection took %.1fs", pose_s)

    # ------------------------------------------------------------------
    # 2. OPTIONAL -- pose key-points JSON (best-effort)
    # ------------------------------------------------------------------
    json_path = Path(task["mask_png"]).with_suffix(".json")
    try:
        tmp_png = json_path.with_suffix(".dwpose.png")
        pts = make_controlnet_mask(task["photo"], str(tmp_png))
        save_keypoints(pts, str(json_path), task["photo"])
        log.debug("Keypoint JSON saved — %s", json_path.name)
    except Exception as e:
        log.warning("pose-JSON skipped: %s", e)

    # ------------------------------------------------------------------
    # 3. DIFFUSION
    # ------------------------------------------------------------------
    gen = torch.Generator(device=device)
    if cfg.get("seed") is not None:
        gen.manual_seed(int(cfg["seed"]))
    actual_seed = gen.initial_seed()

    w, h = src_img.size
    # Clamp generation size: long side between 768 (SDXL quality floor) and 1024 (native).
    # Small source photos are upscaled to the floor so SDXL doesn't generate at sub-768px.
    _MAX_DIM = 1024
    _MIN_DIM = 768
    _long = max(w, h)
    _scale = max(_MIN_DIM / _long, min(1.0, _MAX_DIM / _long))
    w64 = max(64, round(w * _scale / 64) * 64)
    h64 = max(64, round(h * _scale / 64) * 64)

    log.info("Generating image — src %dx%d → %dx%d  steps=%d  seed=%d  (compile warmup on first run)",
             w, h, w64, h64, steps, actual_seed)
    log.debug("VRAM before generation: %s", _vram())
    t_gen = time.time()

    result = pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg.get("neg_prompt"),
        image=control_img,
        num_inference_steps=steps,
        guidance_scale=cfg["guidance"],
        controlnet_conditioning_scale=cfg["cond_scale"],
        generator=gen,
        width=w64,
        height=h64,
    ).images[0]

    gen_s = time.time() - t_gen
    log.info("Generation complete — %.1fs  (%.1fs/step)", gen_s, gen_s / steps)
    log.debug("VRAM after generation: %s", _vram())

    meta = PngImagePlugin.PngInfo()
    meta.add_text("source", os.path.abspath(task["photo"]))
    meta.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in cfg.items():
        meta.add_text(f"cfg/{k}", str(v))
    Path(task["output_png"]).parent.mkdir(parents=True, exist_ok=True)
    result.save(task["output_png"], pnginfo=meta)
    log.info("Saved → %s", Path(task["output_png"]).resolve())

    metrics = {
        "timestamp"     : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_photo"   : str(photo),
        "source_sha256" : source_sha,
        "output_png"    : task["output_png"],
        "seed"          : actual_seed,
        "steps"         : steps,
        "guidance"      : cfg["guidance"],
        "cond_scale"    : cfg["cond_scale"],
        "prompt"        : cfg["prompt"],
        "neg_prompt"    : cfg.get("neg_prompt"),
        "width"         : w64,
        "height"        : h64,
        "profile"       : cfg.get("profile"),
        "timing"        : {
            "pose_s"    : round(pose_s, 2),
            "generate_s": round(gen_s, 2),
            "total_s"   : round(pose_s + gen_s, 2),
            "s_per_step": round(gen_s / steps, 2),
        },
        "rating"        : None,
        "notes"         : None,
    }
    metrics_path = Path(task["output_png"]).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics → %s", metrics_path.resolve())

    # Best-effort DB insert — never blocks or crashes the pipeline
    try:
        _db_root = Path(task["output_png"]).parent
        while _db_root.parent != _db_root and not (_db_root / "yogamann.db").exists():
            _db_root = _db_root.parent
        _db_path = _db_root / "yogamann.db"
        _conn = open_db(_db_path)
        ingest_json(_conn, metrics_path)
        _conn.close()
        log.debug("DB updated — %s", _db_path.name)
    except Exception as _e:
        log.debug("DB write skipped: %s", _e)

    # ------------------------------------------------------------------
    # 4. CONTACT-SHEETS
    # ------------------------------------------------------------------
    try:
        stem  = Path(task["output_png"]).with_suffix("")
        scale = float(cfg.get("sheet_scale", 1.0))
        seed  = cfg.get("seed")
        log.info("Building contact sheets")
        orig = src_img  # already exif_transpose'd at top of render()
        build_comparison_2(
            orig, result,
            stem.with_name(stem.name + "-comparison_2.png"),
            seed, scale,
        )
        build_comparison_4(
            orig, control_img.convert("RGB"), result,
            stem.with_name(stem.name + "-comparison_4.png"),
            seed, scale,
        )
        log.debug("Contact sheets saved → %s  |  %s",
                  stem.with_name(stem.name + "-comparison_2.png"),
                  stem.with_name(stem.name + "-comparison_4.png"))
    except Exception as e:
        log.warning("contact-sheet skipped: %s", e)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--from-worklist", help="JSON file or '-' for stdin")
    cli.add_argument("photo", nargs="?")
    cli.add_argument("mask_png", nargs="?")
    cli.add_argument("output_png", nargs="?")
    cli.add_argument("--steps", type=int)
    cli.add_argument("--cond-scale", dest="cond_scale", type=float, default=1.0)
    cli.add_argument("--guidance", type=float, default=5.0)
    cli.add_argument("--prompt", default="wooden artist mannequin, neutral backdrop")
    cli.add_argument("--neg-prompt")
    cli.add_argument("--seed", type=int)
    cli.add_argument("--meta", default="{}")
    args = cli.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = build_pipe(device)

    if args.from_worklist:
        src = sys.stdin.read() if args.from_worklist == "-" else Path(args.from_worklist).read_text()
        tasks: List[Dict] = json.loads(src)
        for t in tasks:
            if "steps" in t["cfg"] and t["cfg"]["steps"] is not None:
                t["cfg"]["steps"] = int(t["cfg"]["steps"])
        failed = []
        for t in tasks:
            try:
                render(t, device, pipe)
            except Exception as e:
                log.error("SKIP %s — %s: %s", Path(t["photo"]).name, type(e).__name__, e)
                failed.append(t["photo"])
        if failed:
            log.warning("%d task(s) failed: %s", len(failed), ", ".join(Path(p).name for p in failed))
        sys.exit(1 if failed else 0)

    single = {
        "photo": args.photo,
        "mask_png": args.mask_png,
        "output_png": args.output_png,
        "cfg": {
            "steps": int(args.steps) if args.steps else 25,
            "cond_scale": args.cond_scale,
            "guidance": args.guidance,
            "prompt": args.prompt,
            "neg_prompt": args.neg_prompt,
            "seed": args.seed,
            **json.loads(args.meta),
        },
    }
    render(single, device, pipe)
