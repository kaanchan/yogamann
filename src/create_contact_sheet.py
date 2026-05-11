"""
create_contact_sheet.py -- build side-by-side comparison images.

Two layouts:
  <stem>-comparison_2.png  vertical stack  (original / mannequin)
  <stem>-comparison_4.png  2x2 grid        (original / dwpose / mannequin / depth)

Depth estimation uses Depth Anything V2 Small via transformers.
Pass `sheet_scale` (float) at runtime -- defaults to 1.0 (100%).
"""

from __future__ import annotations
import logging
import time
from pathlib import Path

from PIL import Image, PngImagePlugin

log = logging.getLogger(__name__)
_DEPTH_PIPE = None


def _meta(seed: int | None) -> PngImagePlugin.PngInfo:
    meta = PngImagePlugin.PngInfo()
    meta.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    if seed is not None:
        meta.add_text("seed", str(seed))
    return meta


def _depth(img: Image.Image) -> Image.Image:
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        log.info("Loading Depth Anything V2 Small...")
        from transformers import pipeline as hf_pipeline
        _DEPTH_PIPE = hf_pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
        log.debug("Depth model ready")
    result = _DEPTH_PIPE(img)
    return result["depth"].convert("RGB")


def build_comparison_2(orig: Image.Image,
                       mannequin: Image.Image,
                       out_path: Path,
                       seed: int | None,
                       scale: float = 1.0) -> None:
    log.debug("Building 2-panel sheet → %s", out_path.name)
    w, h = orig.size
    w2, h2 = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (w2, h2 * 2), (0, 0, 0))
    canvas.paste(orig.resize((w2, h2), Image.LANCZOS), (0, 0))
    canvas.paste(mannequin.resize((w2, h2), Image.LANCZOS), (0, h2))
    canvas.save(out_path, pnginfo=_meta(seed))


def build_comparison_4(orig: Image.Image,
                       control: Image.Image,
                       mannequin: Image.Image,
                       out_path: Path,
                       seed: int | None,
                       scale: float = 1.0) -> None:
    log.debug("Building 4-panel sheet → %s", out_path.name)
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
