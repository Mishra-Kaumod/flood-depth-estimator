# pipeline/depth.py
"""
Stage 3 — Depth Anything V2
=============================
Input : RGB image (H×W×3 numpy uint8)
Output: depth_map (H×W float32, relative 0-1 or metric metres)

Model: depth-anything/Depth-Anything-V2-Small  (swap for Large when GPU available)

Install:  pip install depth-anything-v2
Weights:  https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

Config flag:  PIPELINE_STUB_MODE=true  → always use gradient stub (local dev)
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

    Args:
        model_path: path to Depth Anything V2 checkpoint (.pth).
                    Ignored when stub_mode=True.
        device:     "cpu" | "cuda" | "mps"
        stub_mode:  Force gradient stub regardless of model_path.
                    Set via PIPELINE_STUB_MODE=true for local dev.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu",
                 stub_mode: bool = False):
        self.device    = device
        self._model    = None
        self._engine   = "gradient_stub"

        if stub_mode:
            log.info("Depth stage: stub_mode=True — skipping model load")
            return

        if model_path and Path(model_path).exists():
            try:
                self._model  = self._load(model_path)
                self._engine = "depth_anything_v2"
                log.info("Depth Anything V2 loaded from %s on %s", model_path, device)
            except ImportError as exc:
                log.warning("Depth model import error (install depth-anything-v2): %s", exc)
            except Exception:
                log.warning("Depth model load failed — using gradient stub", exc_info=True)
        else:
            log.info("Depth Anything V2: no weights path — using gradient stub")

    # ── Public ────────────────────────────────────────────────────────────────
    def predict(self, image_bgr: np.ndarray) -> DepthResult:
        if self._model is not None:
            return self._depth_predict(image_bgr)
        return self._stub_predict(image_bgr)

    # ── Real model ────────────────────────────────────────────────────────────
    def _load(self, path: str):
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise ImportError(
                "depth_anything_v2 package not found. "
                "Install with: pip install depth-anything-v2"
            ) from exc
        import torch
        # encoder='vitl' matches the standard Depth Anything V2 Large config;
        # swap to 'vits' for the Small checkpoint.
        model = DepthAnythingV2(
            encoder="vitl", features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        state = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        return model

    def _depth_predict(self, image_bgr: np.ndarray) -> DepthResult:
        import torch
        rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        depth = self._model.infer_image(rgb)   # H×W numpy float32
        d_min, d_max = depth.min(), depth.max()
        depth_norm = ((depth - d_min) / (d_max - d_min + 1e-6)).astype(np.float32)
        return DepthResult(depth_map=depth_norm, is_metric=False, engine="depth_anything_v2")

    # ── Gradient stub — deeper at bottom (simulates ground-level camera) ──────
    def _stub_predict(self, image_bgr: np.ndarray) -> DepthResult:
        h, w = image_bgr.shape[:2]
        # Vertical gradient: top=0 (far), bottom=1 (near/ground)
        gradient  = np.linspace(0, 1, h, dtype=np.float32)
        depth_map = np.tile(gradient[:, np.newaxis], (1, w))
        # Add slight noise to simulate real depth variation
        noise     = np.random.uniform(-0.05, 0.05, depth_map.shape).astype(np.float32)
        depth_map = np.clip(depth_map + noise, 0, 1)
        return DepthResult(depth_map=depth_map, is_metric=False, engine="gradient_stub")
