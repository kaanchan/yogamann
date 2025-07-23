"""
create_contact_sheet.py – utilities to build side‑by‑side images that let you
quickly judge pose‑to‑mannequin accuracy.

Two layouts are supported:
───────────────────────────────────────────────────────────────────────────────
1. <stem>-comparison_2.png  (vertical stack)
   ┌──────────┐
   │ original │   scaled to `sheet_scale`
   ├──────────┤
   │ mannequin│   scaled to `sheet_scale`
   └──────────┘

2. <stem>-comparison_4.png  (2 × 2 grid)
   ┌────────────┬────────────┐
   │ original   │ controlNet │   each tile scaled so the *whole* sheet width
   ├────────────┼────────────┤   =      original_W × sheet_scale
   │ mannequin  │ depth map  │
   └────────────┴────────────┘

Pass `sheet_scale` (float) at runtime – defaults to 1.0 (100 %).

Dependencies: Pillow, controlnet_aux (for MiDaS depth)
"""

from __future__ import annotations
import time, pathlib
from PIL import Image, PngImagePlugin
from controlnet_aux import MidasDetector

# MiDaS depth model (lazy-loaded once)
_DEPTH: MidasDetector | None = None


def _meta(seed: int | None) -> PngImagePlugin.PngInfo:
    """PNG metadata helper."""
    meta = PngImagePlugin.PngInfo()
    meta.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    if seed is not None:
        meta.add_text("seed", str(seed))
    return meta


def _depth(img: Image.Image) -> Image.Image:
    global _DEPTH
    if _DEPTH is None:
        _DEPTH = MidasDetector.from_pretrained("lllyasviel/ControlNet")
    return _DEPTH(img).convert("RGB")


# ────────────────────────── public API ──────────────────────────────────
def build_comparison_2(orig: Image.Image,
                       mannequin: Image.Image,
                       out_path: pathlib.Path,
                       seed: int | None,
                       scale: float = 1.0) -> None:
    """Vertical stack at `scale`."""
    w, h = orig.size
    w2, h2 = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (w2, h2 * 2), (0, 0, 0))
    canvas.paste(orig.resize((w2, h2), Image.LANCZOS), (0, 0))
    canvas.paste(mannequin.resize((w2, h2), Image.LANCZOS), (0, h2))
    canvas.save(out_path, pnginfo=_meta(seed))


def build_comparison_4(orig: Image.Image,
                       control: Image.Image,
                       mannequin: Image.Image,
                       out_path: pathlib.Path,
                       seed: int | None,
                       scale: float = 1.0) -> None:
    """2×2 grid at `scale`."""
    w, h = orig.size
    w4, h4 = round(w * scale / 2), round(h * scale / 2)
    o = orig.resize((w4, h4), Image.LANCZOS)
    c = control.resize((w4, h4), Image.LANCZOS)
    m = mannequin.resize((w4, h4), Image.LANCZOS)
    d = _depth(orig).resize((w4, h4), Image.LANCZOS)

    canvas = Image.new("RGB", (w4 * 2, h4 * 2), (0, 0, 0))
    canvas.paste(o, (0, 0))
    canvas.paste(c, (w4, 0))
    canvas.paste(m, (0, h4))
    canvas.paste(d, (w4, h4))
    canvas.save(out_path, pnginfo=_meta(seed))
