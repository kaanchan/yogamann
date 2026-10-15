"""
Object Keypoint Similarity (OKS) between two body poses.

Scoring primitive for the pose self-evaluation loop (#16): compare keypoints
re-estimated from a generated image against the source pose.  Pure numpy, no
model or GPU needed.

Formula (COCO): OKS = mean over visible keypoints of exp(-d^2 / (2 s^2 k^2)),
with d the pixel distance, s the object scale (sqrt of its area) and k = 2 * sigma.
"""
import numpy as np

__all__ = ["COCO_SIGMAS", "oks"]

# Per-keypoint falloff for the 17 COCO body keypoints (nose .. right ankle).
COCO_SIGMAS = np.array(
    [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]
) / 10.0

_N_BODY = len(COCO_SIGMAS)


def oks(pred, gt, scale: float, visible=None) -> float:
    """OKS in [0, 1] between predicted and ground-truth keypoints.

    pred, gt: (N, 2) pixel coordinates, N >= 17.  Only the first 17 (COCO body)
    are used, so rtmlib's 133-point WholeBody output can be passed as is.
    scale:    object scale in pixels (sqrt of the person's area); must be > 0.
    visible:  optional bool mask over the body keypoints; invisible ones are
              ignored.  Returns 0.0 when no keypoint is visible.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 2 or pred.shape[0] < _N_BODY:
        raise ValueError(f"expected matching (N>=17, 2) arrays, got {pred.shape} and {gt.shape}")

    pred, gt = pred[:_N_BODY], gt[:_N_BODY]
    mask = np.ones(_N_BODY, dtype=bool) if visible is None else np.asarray(visible, dtype=bool)[:_N_BODY]
    if not mask.any():
        return 0.0

    d2 = np.sum((pred - gt) ** 2, axis=1)
    k = 2.0 * COCO_SIGMAS
    return float(np.mean(np.exp(-d2[mask] / (2.0 * scale ** 2 * k[mask] ** 2))))
