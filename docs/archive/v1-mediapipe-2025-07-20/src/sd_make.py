"""
sd_make.py – Pose mask ➜ mannequin render  (Stable‑Diffusion v1.5 + ControlNet)

• Receives full provenance via --meta JSON
• --dry-run flag: build pipeline, run ONE 8×8 dummy step, exit quick
"""

import os, sys, time, json, argparse
from math import ceil
from PIL import Image, PngImagePlugin
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector

BASE_ID     = "runwayml/stable-diffusion-v1-5"
CONTROL_ID  = "lllyasviel/sd-controlnet-openpose"
DETECTOR_ID = "lllyasviel/ControlNet"


def build_pipeline(device):
    controlnet = ControlNetModel.from_pretrained(CONTROL_ID,
                                                 torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_ID, controlnet=controlnet,
        torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.safety_checker = lambda images, *a, **kw: (images, [False]*len(images))
    return pipe


def run_pipeline(photo, mask_png, out_png,
                 steps, seed,
                 cond_scale, guidance,
                 prompt, neg_prompt,
                 meta_dict, dry_run):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = build_pipeline(device)

    if dry_run:
        print("💡 dry‑run: building pipeline done")
        dummy = torch.zeros(1, 3, 8, 8, device=device)
        pipe(prompt="test", image=dummy, num_inference_steps=1,
             guidance_scale=guidance,
             controlnet_conditioning_scale=cond_scale)
        print("✓ dry‑run OK; no image rendered")
        return

    # 1️⃣ OpenPose mask
    control_img = OpenposeDetector.from_pretrained(DETECTOR_ID)(
        Image.open(photo))
    control_img.save(mask_png)

    # 2️⃣ Full render
    gen = torch.Generator(device).manual_seed(seed) if seed else None
    w, h = Image.open(photo).size
    w8, h8 = ceil(w/8)*8, ceil(h/8)*8

    result = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=control_img,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=cond_scale,
        generator=gen,
        width=w8, height=h8,
    ).images[0]

    # 3️⃣ metadata
    info = PngImagePlugin.PngInfo()
    info.add_text("source", os.path.abspath(photo))
    info.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    if seed is not None:
        info.add_text("seed", str(seed))
    for k, v in meta_dict.items():
        info.add_text(f"cfg/{k}", str(v))
    result.save(out_png, pnginfo=info)


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("photo")
    p.add_argument("mask_png")
    p.add_argument("output_png")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--cond-scale", type=float, required=True)
    p.add_argument("--guidance", type=float, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--neg-prompt")
    p.add_argument("--meta", default="{}")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    run_pipeline(a.photo, a.mask_png, a.output_png,
                 a.steps, a.seed,
                 a.cond_scale, a.guidance,
                 a.prompt, a.neg_prompt,
                 json.loads(a.meta), a.dry_run)
