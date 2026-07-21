# pipeline/segformer.py
"""
Stage 1 — SegFormer Water Segmentation
========================================
Input : RGB image (H×W×3 numpy uint8)
Output: water_mask (H×W numpy bool), water_coverage_pct (float)

Swap `_stub_predict` for real SegFormer weights when available.
Model: nvidia/segformer-b2-finetuned-ade-512-512  (or your fine-tuned checkpoint)
"""

import logging
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("pipeline.segformer")


@dataclass
class SegFormerResult:
    water_mask:         np.ndarray   # H×W bool
    water_coverage_pct: float        # 0-100
    engine:             str          # "segformer" | "heuristic"


class SegFormerStage:
    """
    Loads SegFormer once at startup.
    Call .predict(image_bgr) for each frame.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = device
        self._model = None
        self._engine = "heuristic"

        if model_path and Path(model_path).exists():
            try:
                self._model = self._load(model_path)
                self._engine = "segformer"
                log.info("SegFormer loaded from %s", model_path)
            except Exception:
                log.warning("SegFormer load failed — using heuristic fallback", exc_info=True)
        else:
            log.info("SegFormer: no weights path — using heuristic stub")

    # ── Public ────────────────────────────────────────────────────────────────
    def predict(self, image_bgr: np.ndarray) -> SegFormerResult:
        if self._model is not None:
            return self._segformer_predict(image_bgr)
        return self._heuristic_predict(image_bgr)

    # ── Real model (uncomment + fill in when weights are ready) ───────────────
    def _load(self, path: str):
        # from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        # import torch
        # processor = SegformerImageProcessor.from_pretrained(path)
        # model     = SegformerForSemanticSegmentation.from_pretrained(path).to(self.device)
        # model.eval()
        # return (processor, model)
        raise NotImplementedError("SegFormer weights not yet configured")

    def _segformer_predict(self, image_bgr: np.ndarray) -> SegFormerResult:
        # processor, model = self._model
        # import torch
        # rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # inputs  = processor(images=rgb, return_tensors="pt").to(self.device)
        # with torch.no_grad():
        #     logits = model(**inputs).logits          # (1, num_classes, H/4, W/4)
        # upsampled = torch.nn.functional.interpolate(
        #     logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False)
        # water_class = 21   # ← your water class index
        # mask = upsampled.argmax(dim=1).squeeze().cpu().numpy() == water_class
        # ...
        raise NotImplementedError

    # ── Heuristic stub (blue/dark pixel detection) ────────────────────────────
    def _heuristic_predict(self, image_bgr: np.ndarray) -> SegFormerResult:
        hsv        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        blue_mask  = cv2.inRange(hsv, np.array([85, 20, 30]),  np.array([135, 255, 255]))
        dark_mask  = cv2.inRange(hsv, np.array([0,  0,  20]),  np.array([180,  60, 150]))
        combined   = cv2.bitwise_or(blue_mask, dark_mask).astype(bool)
        # Only consider lower 60 % of frame (road / ground level)
        combined[:int(image_bgr.shape[0] * 0.4), :] = False
        pct = combined.sum() / combined.size * 100
        return SegFormerResult(
            water_mask=combined, water_coverage_pct=round(pct, 2), engine="heuristic"
        )
