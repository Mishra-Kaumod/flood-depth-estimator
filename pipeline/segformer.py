# pipeline/segformer.py
"""
Stage 1 — SegFormer Water Segmentation
========================================
Input : RGB image (H×W×3 numpy uint8)
Output: water_mask (H×W numpy bool), water_coverage_pct (float)

Model: nvidia/segformer-b2-finetuned-ade-512-512  (or your fine-tuned checkpoint)

Install:  pip install transformers torch
Weights:  huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512
          or point to a local HF-format directory with your fine-tuned checkpoint.

Config flag:  PIPELINE_STUB_MODE=true  → always use heuristic (local dev, no GPU)

Water class index:
  ADE20k class 21 = "water" (default).
  Override SEGFORMER_WATER_CLASS=<index> in .env when using a custom checkpoint.
"""

import logging
import os
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("pipeline.segformer")

# Override via env SEGFORMER_WATER_CLASS if your fine-tuned model differs
_WATER_CLASS_INDEX = int(os.environ.get("SEGFORMER_WATER_CLASS", "21"))


@dataclass
class SegFormerResult:
    water_mask:         np.ndarray   # H×W bool
    water_coverage_pct: float        # 0-100
    engine:             str          # "segformer" | "heuristic"


class SegFormerStage:
    """
    Loads SegFormer once at startup.
    Call .predict(image_bgr) for each frame.

    Args:
        model_path: local directory or HF model name/path.
                    Ignored when stub_mode=True.
        device:     "cpu" | "cuda" | "mps"
        stub_mode:  Force heuristic regardless of model_path.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu",
                 stub_mode: bool = False):
        self.device = device
        self._model  = None
        self._engine = "heuristic"

        if stub_mode:
            log.info("SegFormer stage: stub_mode=True — skipping model load")
            return

        if model_path and Path(model_path).exists():
            try:
                self._model  = self._load(model_path)
                self._engine = "segformer"
                log.info("SegFormer loaded from %s on %s", model_path, device)
            except ImportError as exc:
                log.warning("SegFormer import error (install transformers): %s", exc)
            except Exception:
                log.warning("SegFormer load failed — using heuristic fallback", exc_info=True)
        else:
            log.info("SegFormer: no weights path — using heuristic stub")

    # ── Public ────────────────────────────────────────────────────────────────
    def predict(self, image_bgr: np.ndarray) -> SegFormerResult:
        if self._model is not None:
            return self._segformer_predict(image_bgr)
        return self._heuristic_predict(image_bgr)

    # ── Real model ────────────────────────────────────────────────────────────
    def _load(self, path: str):
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers package not found. "
                "Install with: pip install transformers"
            ) from exc
        import torch
        processor = SegformerImageProcessor.from_pretrained(path)
        model     = SegformerForSemanticSegmentation.from_pretrained(path)
        model.to(self.device).eval()
        return (processor, model)

    def _segformer_predict(self, image_bgr: np.ndarray) -> SegFormerResult:
        import torch
        processor, model = self._model
        rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits          # (1, num_classes, H/4, W/4)
        upsampled = torch.nn.functional.interpolate(
            logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False
        )
        mask = upsampled.argmax(dim=1).squeeze().cpu().numpy() == _WATER_CLASS_INDEX
        mask = mask.astype(bool)
        pct  = float(mask.sum()) / mask.size * 100
        return SegFormerResult(
            water_mask=mask, water_coverage_pct=round(pct, 2), engine="segformer"
        )

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
