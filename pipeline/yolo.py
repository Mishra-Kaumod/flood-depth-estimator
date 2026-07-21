# pipeline/yolo.py
"""
Stage 2 — YOLOv8 Reference Object Detection
=============================================
Input : RGB image (H×W×3 numpy uint8)
Output: list of ReferenceObject with known real-world heights

Reference heights are used by the Fusion Engine to calibrate the
depth map from relative units to absolute centimetres.

Known heights (cm):
  car door    ~120  |  stop sign   ~210  |  person      ~170
  truck       ~380  |  fire hydrant ~60  |  traffic cone ~70
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

log = logging.getLogger("pipeline.yolo")

# Known real-world heights in cm for calibration
REFERENCE_HEIGHTS_CM: dict[str, float] = {
    "car":          120.0,
    "truck":        380.0,
    "person":       170.0,
    "stop sign":    210.0,
    "fire hydrant":  60.0,
    "traffic cone":  70.0,
    "bicycle":      100.0,
    "motorcycle":   110.0,
}


@dataclass
class ReferenceObject:
    class_name:       str
    confidence:       float
    bbox_xyxy:        List[int]        # [x1,y1,x2,y2] in pixels
    real_height_cm:   float            # known height
    pixel_height:     int              # bbox height in pixels
    depth_scale:      float = 0.0      # real_height / pixel_height (filled by fusion)


@dataclass
class YOLOResult:
    objects:   List[ReferenceObject] = field(default_factory=list)
    engine:    str = "stub"


class YOLOStage:
    """
    Detects reference objects for depth calibration.
    Loads YOLOv8 once at startup.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu",
                 conf_threshold: float = 0.4):
        self.device    = device
        self.conf_thr  = conf_threshold
        self._model    = None
        self._engine   = "stub"

        if model_path and Path(model_path).exists():
            try:
                self._model = self._load(model_path)
                self._engine = "yolov8"
                log.info("YOLOv8 loaded from %s", model_path)
            except Exception:
                log.warning("YOLOv8 load failed — using stub", exc_info=True)
        else:
            log.info("YOLOv8: no weights path — using stub")

    # ── Public ────────────────────────────────────────────────────────────────
    def predict(self, image_bgr: np.ndarray) -> YOLOResult:
        if self._model is not None:
            return self._yolo_predict(image_bgr)
        return self._stub_predict(image_bgr)

    # ── Real model ────────────────────────────────────────────────────────────
    def _load(self, path: str):
        from ultralytics import YOLO
        return YOLO(path)

    def _yolo_predict(self, image_bgr: np.ndarray) -> YOLOResult:
        results = self._model(image_bgr, conf=self.conf_thr, verbose=False)[0]
        objects = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            if cls_name not in REFERENCE_HEIGHTS_CM:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            objects.append(ReferenceObject(
                class_name     = cls_name,
                confidence     = float(box.conf),
                bbox_xyxy      = [x1, y1, x2, y2],
                real_height_cm = REFERENCE_HEIGHTS_CM[cls_name],
                pixel_height   = y2 - y1,
            ))
        return YOLOResult(objects=objects, engine="yolov8")

    # ── Stub (no detection, fusion will use fallback calibration) ─────────────
    def _stub_predict(self, image_bgr: np.ndarray) -> YOLOResult:
        h = image_bgr.shape[0]
        # Synthetic car-sized reference at bottom-centre for calibration stub
        return YOLOResult(
            objects=[ReferenceObject(
                class_name     = "car",
                confidence     = 0.0,
                bbox_xyxy      = [image_bgr.shape[1]//4, int(h*0.6),
                                  image_bgr.shape[1]*3//4, int(h*0.9)],
                real_height_cm = 120.0,
                pixel_height   = int(h * 0.3),
            )],
            engine="stub",
        )
