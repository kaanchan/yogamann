"""
ONNX-based DWPose detector — drop-in for controlnet_aux.DWposeDetector.

Uses rtmlib.Wholebody (onnxruntime) for inference; the rendering pipeline
is the pure-numpy code from controlnet_aux.dwpose (no mmcv required).
"""
import cv2
import numpy as np
import torch
from PIL import Image
from rtmlib import Wholebody as RTMWholebody

from controlnet_aux.dwpose import draw_pose
from controlnet_aux.util import HWC3, resize_image


class DWposeDetector:
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = RTMWholebody(backend="onnxruntime", device=device, to_openpose=False)

    def to(self, device: str) -> "DWposeDetector":
        self._model = RTMWholebody(backend="onnxruntime", device=device, to_openpose=False)
        return self

    def __call__(
        self,
        input_image,
        detect_resolution: int = 512,
        image_resolution: int = 512,
        output_type: str = "pil",
        **kwargs,
    ):
        img = cv2.cvtColor(np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        img = HWC3(img)
        img = resize_image(img, detect_resolution)
        H, W, _ = img.shape

        keypoints, scores = self._model(img)  # (N, 133, 2), (N, 133)

        if keypoints is None or len(keypoints) == 0:
            detected_map = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            detected_map = self._render(keypoints, scores, H, W)

        detected_map = HWC3(detected_map)
        out = resize_image(img, image_resolution)
        H2, W2, _ = out.shape
        detected_map = cv2.resize(detected_map, (W2, H2), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            return Image.fromarray(detected_map)
        return detected_map

    @staticmethod
    def _render(keypoints: np.ndarray, scores: np.ndarray, H: int, W: int) -> np.ndarray:
        """Convert rtmlib 133-pt output to an OpenPose skeleton image.

        Replicates controlnet_aux Wholebody post-processing:
          1. Compute neck as mean of left/right shoulder (index 5, 6).
          2. Insert neck at index 17 → 134 keypoints.
          3. Remap MMPose body indices to OpenPose body indices.
        """
        visible = (scores > 0.3).astype(float)
        # (N, 133, 4): x, y, score, visible
        kp_info = np.concatenate([keypoints, scores[..., None], visible[..., None]], axis=-1)

        # Neck = mean of left (5) and right (6) shoulder
        neck = np.mean(kp_info[:, [5, 6]], axis=1, keepdims=True)  # (N, 1, 4)
        neck[:, :, 2:4] = np.logical_and(
            kp_info[:, 5:6, 2:4] > 0.3,
            kp_info[:, 6:7, 2:4] > 0.3,
        ).astype(float)
        # Insert neck at index 17 → (N, 134, 4)
        kp_info = np.concatenate([kp_info[:, :17], neck, kp_info[:, 17:]], axis=1)

        # MMPose body indices → OpenPose body indices
        mmpose_idx  = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        remapped = kp_info.copy()
        remapped[:, openpose_idx] = kp_info[:, mmpose_idx]
        kp_info = remapped

        candidate = kp_info[..., :2].copy()   # (N, 134, 2)  pixel coords
        subset    = kp_info[..., 2].copy()    # (N, 134)     scores

        nums = candidate.shape[0]
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)

        body  = candidate[:, :18].reshape(nums * 18, 2)
        score = subset[:, :18].copy()
        for i in range(nums):
            for j in range(18):
                score[i, j] = int(18 * i + j) if score[i, j] > 0.3 else -1

        candidate[subset < 0.3] = -1

        faces = candidate[:, 24:92]
        hands = np.vstack([candidate[:, 92:113], candidate[:, 113:]])

        pose = dict(bodies=dict(candidate=body, subset=score), hands=hands, faces=faces)
        return draw_pose(pose, H, W)
