"""
extract_pose.py -- detect a full-body pose and export:

1. Transparent PNG  (DWPose skeleton, ControlNet-ready)
2. JSON             (raw pixel coordinates + provenance metadata)

Uses DWposeDetector from controlnet_aux (ONNX-based, no mediapipe).
"""
import json
import logging
import os
import time
from typing import List, Tuple

from PIL import Image, PngImagePlugin
from dwpose_onnx import DWposeDetector

log = logging.getLogger(__name__)

__all__ = ["make_controlnet_mask", "save_keypoints"]

CANVAS_SIZE  = 768

_DETECTOR: DWposeDetector | None = None


def _get_detector() -> DWposeDetector:
    global _DETECTOR
    if _DETECTOR is None:
        log.info("Loading DWPose detector (downloading weights on first run)")
        _DETECTOR = DWposeDetector()
        log.debug("DWPose detector ready")
    return _DETECTOR


def make_controlnet_mask(img_path: str,
                         out_png: str,
                         size: int = CANVAS_SIZE) -> List[Tuple[int, int]]:
    """
    Run DWPose on `img_path`, write the skeleton PNG to `out_png`.
    Returns an empty list (DWPose returns a rendered image, not raw keypoints).
    Raises RuntimeError if detection fails.
    """
    detector = _get_detector()
    log.debug("Running DWPose on %s", os.path.basename(img_path))
    img = Image.open(img_path).convert("RGB")
    skeleton = detector(img)
    if skeleton is None:
        raise RuntimeError(f"DWPose found no pose in {img_path}")

    skeleton = skeleton.resize((size, size), Image.LANCZOS)

    meta = PngImagePlugin.PngInfo()
    meta.add_text("source",    os.path.abspath(img_path))
    meta.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    skeleton.save(out_png, pnginfo=meta)

    return []


def save_keypoints(pts: List[Tuple[int, int]], out_json: str, img_path: str):
    """Write a provenance JSON sidecar (keypoints list may be empty for DWPose)."""
    payload = {
        "source":    os.path.abspath(img_path),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector":  "DWPose",
        "keypoints": pts,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
