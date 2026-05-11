"""
sd_make.py – worker (single task or work‑list)
· Swaps PNDM → DPMSolverMultistep
· Safe on Windows diffusers 0.34
· 2025‑07‑20:  ➜  writes ControlNet JSON & contact‑sheets
"""

from __future__ import annotations
import argparse, json, os, sys, time, warnings
from math import ceil
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, PngImagePlugin
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from controlnet_aux import OpenposeDetector

# new ───────────────────────────────────────────────────────────────────
from extract_pose import make_controlnet_mask, save_keypoints
from create_contact_sheet import build_comparison_2, build_comparison_4
# ───────────────────────────────────────────────────────────────────────

BASE_ID     = "runwayml/stable-diffusion-v1-5"
CONTROL_ID  = "lllyasviel/sd-controlnet-openpose"
DETECTOR_ID = "lllyasviel/ControlNet"

# ── helper for schedulers that lack .to() (Windows wheels) ────────────
def _move_sched_tensors(sched, device):
    for k, v in list(sched.__dict__.items()):
        if torch.is_tensor(v):
            sched.__dict__[k] = v.to(device)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            sched.__dict__[k] = [t.to(device) for t in v]


# ── build the pipeline once per worker process ────────────────────────
def build_pipe(device: str):
    cn = ControlNetModel.from_pretrained(CONTROL_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_ID,
        controlnet=cn,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    pipe = pipe.to(device)
    # PyTorch 2.6+ uses built-in SDPA (Flash Attention) automatically — faster than
    # xformers on Blackwell and requires no separate install.
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    return pipe


# ── render ONE task dict ───────────────────────────────────────────────
def render(task: Dict, device: str, pipe) -> None:
    cfg = task["cfg"]
    steps = int(cfg.get("steps") or 50)

    print(
        f"[worker] -> {Path(task['output_png']).name} (steps={steps})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 1. CONTROLNET MASK  (OpenPose)
    # ------------------------------------------------------------------
    control_img = OpenposeDetector.from_pretrained(DETECTOR_ID)(
        Image.open(task["photo"])
    )
    control_img.save(task["mask_png"])

    # ------------------------------------------------------------------
    # 2. OPTIONAL – Pose key‑points JSON via MediaPipe
    # ------------------------------------------------------------------
    json_path = Path(task["mask_png"]).with_suffix(".json")
    try:
        tmp_png = json_path.with_suffix(".mediapipe.png")
        pts = make_controlnet_mask(task["photo"], str(tmp_png))
        save_keypoints(pts, str(json_path), task["photo"])
    except Exception as e:  # pose might fail on very wide crops
        print(f"⚠ pose‑JSON skipped: {e}")

    # ------------------------------------------------------------------
    # 3. DIFFUSION
    # ------------------------------------------------------------------
    _move_sched_tensors(pipe.scheduler, "cpu")  # safety for NumPy ops

    gen = torch.Generator(device=device)
    if cfg.get("seed") is not None:
        gen.manual_seed(int(cfg["seed"]))

    w, h = Image.open(task["photo"]).size
    w8, h8 = max(512, ceil(w / 8) * 8), max(512, ceil(h / 8) * 8)

    result = pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg["neg_prompt"],
        image=control_img,
        num_inference_steps=steps,
        guidance_scale=cfg["guidance"],
        controlnet_conditioning_scale=cfg["cond_scale"],
        generator=gen,
        width=w8,
        height=h8,
    ).images[0]

    meta = PngImagePlugin.PngInfo()
    meta.add_text("source", os.path.abspath(task["photo"]))
    meta.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in cfg.items():
        meta.add_text(f"cfg/{k}", str(v))
    Path(task["output_png"]).parent.mkdir(parents=True, exist_ok=True)
    result.save(task["output_png"], pnginfo=meta)
    print("✓", Path(task["output_png"]).name)

    # ------------------------------------------------------------------
    # 4. CONTACT‑SHEETS
    # ------------------------------------------------------------------
    try:
        stem = Path(task["output_png"]).with_suffix("")
        scale = float(cfg.get("sheet_scale", 1.0))
        seed  = cfg.get("seed")

        orig = Image.open(task["photo"]).convert("RGB")
        build_comparison_2(
            orig,
            result,
            stem.with_name(stem.name + "-comparison_2.png"),
            seed,
            scale,
        )

        build_comparison_4(
            orig,
            control_img.convert("RGB"),
            result,
            stem.with_name(stem.name + "-comparison_4.png"),
            seed,
            scale,
        )
    except Exception as e:
        print(f"⚠ contact‑sheet skipped: {e}")


# ── CLI boilerplate (unchanged) ────────────────────────────────────────
if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--from-worklist", help="JSON file or '-' for stdin")
    cli.add_argument("photo", nargs="?")
    cli.add_argument("mask_png", nargs="?")
    cli.add_argument("output_png", nargs="?")
    cli.add_argument("--steps", type=int)
    cli.add_argument("--cond-scale", dest="cond_scale", type=float, default=1.0)
    cli.add_argument("--guidance", type=float, default=7.5)
    cli.add_argument("--prompt", default="wooden artist mannequin, neutral backdrop")
    cli.add_argument("--neg-prompt")
    cli.add_argument("--seed", type=int)
    cli.add_argument("--meta", default="{}")
    args = cli.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = build_pipe(device)

    # work‑list mode -----------------------------------------------------
    if args.from_worklist:
        src = sys.stdin.read() if args.from_worklist == "-" else Path(args.from_worklist).read_text()
        tasks: List[Dict] = json.loads(src)
        for t in tasks:
            if "steps" in t["cfg"] and t["cfg"]["steps"] is not None:
                t["cfg"]["steps"] = int(t["cfg"]["steps"])
        for t in tasks:
            render(t, device, pipe)
        sys.exit()

    # single‑image fallback ---------------------------------------------
    single = {
        "photo": args.photo,
        "mask_png": args.mask_png,
        "output_png": args.output_png,
        "cfg": {
            "steps": int(args.steps) if args.steps else 50,
            "cond_scale": args.cond_scale,
            "guidance": args.guidance,
            "prompt": args.prompt,
            "neg_prompt": args.neg_prompt,
            "seed": args.seed,
            **json.loads(args.meta),
        },
    }
    render(single, device, pipe)
