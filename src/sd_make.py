"""
sd_make.py – worker (single task or work‑list)
· Swaps PNDM → DPMSolverMultistep
· Safe on Windows diffusers 0.34
"""

import warnings, json, argparse, os, sys, time
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

# ── silence ControlNet/Tiny‑ViT re‑registration spam ────────────────
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting tiny_vit_.* in registry",
    category=UserWarning,
    module=r"controlnet_aux\.segment_anything\.modeling\.tiny_vit_sam",
)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_ID     = "runwayml/stable-diffusion-v1-5"
CONTROL_ID  = "lllyasviel/sd-controlnet-openpose"
DETECTOR_ID = "lllyasviel/ControlNet"


# ── helper for schedulers that lack .to() (Windows wheels) ───────────
def _move_sched_tensors(sched, device):
    for k, v in list(sched.__dict__.items()):
        if torch.is_tensor(v):
            sched.__dict__[k] = v.to(device)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            sched.__dict__[k] = [t.to(device) for t in v]


# ── build the pipeline once per worker process ───────────────────────
def build_pipe(device: str):
    cn = ControlNetModel.from_pretrained(CONTROL_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_ID, controlnet=cn,
        torch_dtype=torch.float16, safety_checker=None
    ).to(device)

    # drop buggy PNDM → use DPMSolver‑Multistep
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    # move scheduler tensors to render device
    if hasattr(pipe.scheduler, "to"):
        pipe.scheduler = pipe.scheduler.to(device)
    else:
        _move_sched_tensors(pipe.scheduler, device)

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    return pipe


# ── render ONE task dict ──────────────────────────────────────────────
def render(task: Dict, device: str, pipe) -> None:
    cfg = task["cfg"]
    
    # Ensure steps is an integer
    steps = int(cfg["steps"]) if cfg["steps"] is not None else 50
    
    print(f"[worker] -> {Path(task['output_png']).name} (steps={steps})",
          flush=True)

    # Pose mask (shared across versions)
    control_img = OpenposeDetector.from_pretrained(DETECTOR_ID)(
        Image.open(task["photo"]))
    control_img.save(task["mask_png"])

    # Scheduler timesteps - handle on CPU to avoid CUDA tensor numpy conversion issues
    # The pipeline will handle timestep setup internally, but we need to ensure
    # the scheduler's internal tensors are properly managed
    _move_sched_tensors(pipe.scheduler, "cpu")

    # Generator / seed
    gen = torch.Generator(device=device)
    if cfg.get("seed") is not None:
        gen.manual_seed(int(cfg["seed"]))

    # Canvas ≥512 px, mult‑of‑8
    w, h = Image.open(task["photo"]).size
    w8, h8 = max(512, ceil(w / 8) * 8), max(512, ceil(h / 8) * 8)

    # Diffusion - use integer steps
    result = pipe(
        prompt                       = cfg["prompt"],
        negative_prompt              = cfg["neg_prompt"],
        image                        = control_img,
        num_inference_steps          = steps,
        guidance_scale               = cfg["guidance"],
        controlnet_conditioning_scale= cfg["cond_scale"],
        generator=gen,
        width=w8, height=h8,
    ).images[0]

    # Metadata
    meta = PngImagePlugin.PngInfo()
    meta.add_text("source", os.path.abspath(task["photo"]))
    meta.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in cfg.items():
        meta.add_text(f"cfg/{k}", str(v))

    Path(task["output_png"]).parent.mkdir(parents=True, exist_ok=True)
    result.save(task["output_png"], pnginfo=meta)
    print("✓", Path(task["output_png"]).name)


# ── CLI boilerplate with type conversion ──────────────────────────────
if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--from-worklist", help="JSON file or '-' for stdin")
    cli.add_argument("photo", nargs="?")
    cli.add_argument("mask_png", nargs="?")
    cli.add_argument("output_png", nargs="?")
    cli.add_argument("--steps", type=int)
    cli.add_argument("--cond-scale", type=float)
    cli.add_argument("--guidance", type=float)
    cli.add_argument("--prompt")
    cli.add_argument("--neg-prompt")
    cli.add_argument("--seed", type=int)
    cli.add_argument("--meta", default="{}")
    args = cli.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = build_pipe(device)

    # batch mode
    if args.from_worklist:
        tasks = (json.loads(sys.stdin.read()) if args.from_worklist == "-"
                 else json.loads(Path(args.from_worklist).read_text()))
        
        # Ensure steps is integer in all tasks
        for t in tasks:
            if "steps" in t["cfg"] and t["cfg"]["steps"] is not None:
                t["cfg"]["steps"] = int(t["cfg"]["steps"])
        
        for t in tasks:
            render(t, device, pipe)
        sys.exit()

    # single task - ensure steps is integer
    single = {
        "photo"     : args.photo,
        "mask_png"  : args.mask_png,
        "output_png": args.output_png,
        "cfg": {
            "steps"      : int(args.steps) if args.steps is not None else 50,
            "cond_scale" : args.cond_scale,
            "guidance"   : args.guidance,
            "prompt"     : args.prompt,
            "neg_prompt" : args.neg_prompt,
            "seed"       : args.seed,
            **json.loads(args.meta)
        }
    }
    render(single, device, pipe)