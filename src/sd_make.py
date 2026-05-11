"""
sd_make.py -- worker (single task or work-list)
· SDXL + ControlNet-OpenPose-SDXL (DWPose-trained)
· PyTorch nightly cu130, bfloat16 (Blackwell native)
· Closes #7
"""

from __future__ import annotations
import argparse, json, os, sys, time, warnings
from math import ceil
from pathlib import Path
from typing import Dict, List

import torch
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


def build_pipe(device: str):
    cn = ControlNetModel.from_pretrained(CONTROL_ID, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_ID,
        controlnet=cn,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    # PyTorch 2.6+ SDPA (Flash Attention 3 on Blackwell) is used automatically.
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    # Blackwell kernel fusion: compiles the UNet after the first forward pass.
    # fullgraph=False tolerates diffusers' graph breaks (dynamic shapes, Python control flow).
    # First generation call triggers compilation (~2-5 min); subsequent calls are fast.
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
    return pipe


def render(task: Dict, device: str, pipe) -> None:
    cfg   = task["cfg"]
    steps = int(cfg.get("steps") or 25)

    print(f"[worker] -> {Path(task['output_png']).name} (steps={steps})", flush=True)

    # ------------------------------------------------------------------
    # 1. CONTROLNET MASK  (DWPose -- no mediapipe)
    # ------------------------------------------------------------------
    detector = DWposeDetector.from_pretrained(DETECTOR_ID)
    control_img = detector(Image.open(task["photo"]).convert("RGB"))
    control_img.save(task["mask_png"])

    # ------------------------------------------------------------------
    # 2. OPTIONAL -- pose key-points JSON (best-effort)
    # ------------------------------------------------------------------
    json_path = Path(task["mask_png"]).with_suffix(".json")
    try:
        tmp_png = json_path.with_suffix(".dwpose.png")
        pts = make_controlnet_mask(task["photo"], str(tmp_png))
        save_keypoints(pts, str(json_path), task["photo"])
    except Exception as e:
        print(f"[warn] pose-JSON skipped: {e}")

    # ------------------------------------------------------------------
    # 3. DIFFUSION
    # ------------------------------------------------------------------
    gen = torch.Generator(device=device)
    if cfg.get("seed") is not None:
        gen.manual_seed(int(cfg["seed"]))

    img = Image.open(task["photo"])
    w, h = img.size
    # SDXL native resolution is 1024; multiples of 64
    w64 = max(1024, ceil(w / 64) * 64)
    h64 = max(1024, ceil(h / 64) * 64)

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

    meta = PngImagePlugin.PngInfo()
    meta.add_text("source", os.path.abspath(task["photo"]))
    meta.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in cfg.items():
        meta.add_text(f"cfg/{k}", str(v))
    Path(task["output_png"]).parent.mkdir(parents=True, exist_ok=True)
    result.save(task["output_png"], pnginfo=meta)
    print("[OK]", Path(task["output_png"]).name)

    # ------------------------------------------------------------------
    # 4. CONTACT-SHEETS
    # ------------------------------------------------------------------
    try:
        stem  = Path(task["output_png"]).with_suffix("")
        scale = float(cfg.get("sheet_scale", 1.0))
        seed  = cfg.get("seed")

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
    except Exception as e:
        print(f"[warn] contact-sheet skipped: {e}")


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
