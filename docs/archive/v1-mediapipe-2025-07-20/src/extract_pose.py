"""
extract_pose.py  – detect a full‑body pose and export:

1. Transparent PNG  (ControlNet‑ready skeleton)
2. JSON             (raw pixel coordinates + provenance metadata)

The MediaPipe "POSE" solution gives us 33 landmarks. We up‑scale /
pad the detected key‑points onto a square canvas so the mask can be
fed directly into the ControlNet‑Pose model.
"""
import cv2
import json
import os
import time
from typing import List, Tuple

from PIL import Image, ImageDraw, PngImagePlugin
import mediapipe as mp

__all__ = ["make_controlnet_mask", "save_keypoints"]

# ── configuration ────────────────────────────────────────────────────────
CANVAS_SIZE = 768                   # square PNG edge in pixels
LINE_CLR    = (220, 220, 220, 255)  # light‑grey RGBA for limb lines
JOINT_CLR   = (220, 220, 220, 255)  # same colour for joint circles
RADIUS_PX   = 6                     # joint marker radius

# initialise once (heavy op); static_image_mode=True = single‑frame pose
mp_pose   = mp.solutions.pose.Pose(static_image_mode=True)
CONNECT   = mp.solutions.pose.POSE_CONNECTIONS


# ── helper: draw lines & dots on a transparent canvas ────────────────────
def _draw_skeleton(landmarks: List[Tuple[int, int]],
                   size: int = CANVAS_SIZE,
                   radius: int = RADIUS_PX) -> Image.Image:
    """Return a transparent RGBA PIL image with the skeleton drawn."""
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw   = ImageDraw.Draw(canvas)

    # draw limbs
    for a, b in CONNECT:
        draw.line([landmarks[a], landmarks[b]], fill=LINE_CLR, width=radius * 2)

    # draw joints
    for x, y in landmarks:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=JOINT_CLR)
    return canvas


# ── public API: create mask + return raw pts ─────────────────────────────
def make_controlnet_mask(img_path: str,
                         out_png: str,
                         size: int = CANVAS_SIZE) -> List[Tuple[int, int]]:
    """
    Detect a single person in `img_path` and write a transparent PNG mask
    to `out_png`. Returns the list of pixel (x,y) pairs for JSON export.
    Raises RuntimeError if no pose is found.
    """
    # 1. run pose detection on the image
    bgr = cv2.imread(img_path)
    results = mp_pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        raise RuntimeError(f"No pose found in {img_path}")

    # 2. map fractional coordinates → absolute pixels on a fixed square canvas
    h,  w  = bgr.shape[:2]
    scale  = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2

    pts = [(
        int(p.x * new_w) + pad_x,
        int(p.y * new_h) + pad_y
    ) for p in results.pose_landmarks.landmark]

    # 3. draw skeleton
    png_img = _draw_skeleton(pts, size=size)

    # 4. embed provenance metadata inside the PNG
    meta = PngImagePlugin.PngInfo()
    meta.add_text("source",    os.path.abspath(img_path))
    meta.add_text("generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    png_img.save(out_png, pnginfo=meta)

    return pts


def save_keypoints(pts: List[Tuple[int, int]], out_json: str, img_path: str):
    """
    Dump the pixel coordinates + provenance to a JSON sidecar file.
    """
    payload = {
        "source":    os.path.abspath(img_path),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "keypoints": pts
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
