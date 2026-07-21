# pipeline/depth.py
"""
Stage 3 — Depth Anything V2
=============================
Input : RGB image (H×W×3 numpy uint8)
Output: depth_map (H×W float32, relative 0-1 or metric metres)

Model: depth-anything/Depth-Anything-V2-Small  (swap for Large when GPU available)
"""

import logging
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("pipeline.depth")


@dataclass
class DepthResult:
    depth_map:    np.ndarray   # H×W float32 (relative 0→1, far→near)
    is_metric:    bool         # True if absolute metres, False if relative
    engine:       str          # "depth_anything_v2" | "gradient_stub"


class DepthStage:
    """
    Produces a dense depth map for every frame.
    Relative map is calibrated to absolute cm in FusionStage.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device  = device
        self._model  = None
        self._engine = "gradient_stub"

        if model_path and Path(model_path).exists():
            try:
                self._model = self._load(model_path)
                self._engine = "depth_anything_v2"
                log.info("Depth Anything V2 loaded from %s", model_path)
            except Exception:
                log.warning("Depth model load failed — using stub", exc_info=True)
        else:
            log.info("Depth Anything V2: no weights — using gradient stub")

    # ── Public ────────────────────────────────────────────────────────────────
    def predict(self, image_bgr: np.ndarray) -> DepthResult:
        if self._model is not None:
            return self._depth_predict(image_bgr)
        return self._stub_predict(image_bgr)

    # ── Real model ────────────────────────────────────────────────────────────
    def _load(self, path: str):
        # from depth_anything_v2.dpt import DepthAnythingV2
        # import torch
        # model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256,512,1024,1024])
        # model.load_state_dict(torch.load(path, map_location=self.device))
        # model.to(self.device).eval()
        # return model
        raise NotImplementedError("Depth Anything V2 weights not configured")

    def _depth_predict(self, image_bgr: np.ndarray) -> DepthResult:
        # import torch
        # rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # depth = self._model.infer_image(rgb)          # returns H×W numpy float32
        # depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        # return DepthResult(depth_map=depth_norm, is_metric=False, engine="depth_anything_v2")
        raise NotImplementedError

    # ── Gradient stub — deeper at bottom (simulates ground-level camera) ──────
    def _stub_predict(self, image_bgr: np.ndarray) -> DepthResult:
        h, w = image_bgr.shape[:2]
        # Vertical gradient: top=0 (far), bottom=1 (near/ground)
        gradient = np.linspace(0, 1, h, dtype=np.float32)
        depth_map = np.tile(gradient[:, np.newaxis], (1, w))
        # Add slight noise to simulate real depth variation
        noise = np.random.uniform(-0.05, 0.05, depth_map.shape).astype(np.float32)
        depth_map = np.clip(depth_map + noise, 0, 1)
        return DepthResult(depth_map=depth_map, is_metric=False, engine="gradient_stub")
