"""
sd_make.py -- worker (single task or work-list)
· SDXL + ControlNet-OpenPose-SDXL (DWPose-trained)
· PyTorch nightly cu130, bfloat16 (Blackwell native)
· Closes #7
"""

from __future__ import annotations
import argparse, json, logging, os, sys, time, warnings
from math import ceil
from pathlib import Path
from typing import Dict, List

import torch

_log_level = os.environ.get("YOGAMANN_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, _log_level),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
from PIL import Image, PngImagePlugin
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from controlnet_aux import DWposeDetector

from extract_pose import make_controlnet_mask, save_keypoints
from create_contact_sheet import build_comparison_2, build_comparison_4

BASE_ID    = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROL_ID = "xinsir/controlnet-openpose-sdxl-1.0"
DETECTOR_ID = "lllyasviel/Annotators"


def _vram() -> str:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        return f"{free/1e9:.1f}/{total/1e9:.1f} GB free"
    return "N/A"


def build_pipe(device: str):
    log.info("Loading ControlNet — %s (bfloat16)", CONTROL_ID)
    cn = ControlNetModel.from_pretrained(CONTROL_ID, torch_dtype=torch.bfloat16)

    log.info("Loading SDXL pipeline — %s (bfloat16)", BASE_ID)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_ID,
        controlnet=cn,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    log.debug("VRAM after pipeline load: %s", _vram())

    # Blackwell kernel fusion. First forward pass triggers compilation (~2-5 min).
    log.info("Compiling UNet with torch.compile (first run: ~2-5 min warmup)")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
    return pipe


def render(task: Dict, device: str, pipe) -> None:
    cfg   = task["cfg"]
    steps = int(cfg.get("steps") or 25)
    photo = Path(task["photo"])

    log.info("render — %s  steps=%d  guidance=%.1f  seed=%s",
             photo.name, steps, cfg["guidance"], cfg.get("seed", "random"))

    # ------------------------------------------------------------------
    # 1. CONTROLNET MASK  (DWPose)
    # ------------------------------------------------------------------
    log.info("Detecting pose — %s", photo.name)
    t0 = time.time()
    detector = DWposeDetector.from_pretrained(DETECTOR_ID)
    control_img = detector(Image.open(task["photo"]).convert("RGB"))
    control_img.save(task["mask_png"])
    log.debug("Pose detection took %.1fs", time.time() - t0)

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

    img = Image.open(task["photo"])
    w, h = img.size
    w64 = max(1024, ceil(w / 64) * 64)
    h64 = max(1024, ceil(h / 64) * 64)

    log.info("Generating image — %dx%d  steps=%d  (compile warmup on first run)",
             w64, h64, steps)
    log.debug("VRAM before generation: %s", _vram())
    t0 = time.time()

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

    elapsed = time.time() - t0
    log.info("Generation complete — %.1fs  (%.1fs/step)", elapsed, elapsed / steps)
    log.debug("VRAM after generation: %s", _vram())

    meta = PngImagePlugin.PngInfo()
    meta.add_text("source", os.path.abspath(task["photo"]))
    meta.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in cfg.items():
        meta.add_text(f"cfg/{k}", str(v))
    Path(task["output_png"]).parent.mkdir(parents=True, exist_ok=True)
    result.save(task["output_png"], pnginfo=meta)
    log.info("Saved → %s", Path(task["output_png"]).name)

    # ------------------------------------------------------------------
    # 4. CONTACT-SHEETS
    # ------------------------------------------------------------------
    try:
        stem  = Path(task["output_png"]).with_suffix("")
        scale = float(cfg.get("sheet_scale", 1.0))
        seed  = cfg.get("seed")
        log.info("Building contact sheets")
        orig = Image.open(task["photo"]).convert("RGB")
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
        log.debug("Contact sheets saved")
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
        for t in tasks:
            render(t, device, pipe)
        sys.exit()

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
