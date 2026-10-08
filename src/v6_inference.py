"""Shared V6 entrypoint contract; no depth decisions belong here.

All deployment surfaces use PIL-decoded uint8 RGB, the same pipeline factory,
and V6ShadowPipeline.predict(). Video must save its frame before using this
loader so image uploads and file inference see the same encoded image.
"""
from io import BytesIO
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_v6_rgb(source: str | Path | bytes) -> np.ndarray:
    """Decode encoded image bytes or a file as RGB without resizing/normalizing."""
    with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as image:
        return np.asarray(image.convert("RGB"))


def create_v6_pipeline():
    """Single model-construction path, with the existing model defaults intact."""
    from .segformer_yolo_depthv2_pipeline import SegformerYoloDepthV2Pipeline
    from .v6_shadow_pipeline import V6ShadowPipeline

    return V6ShadowPipeline(SegformerYoloDepthV2Pipeline())


def finite_depth(value: Any) -> float | None:
    """Serialization only: retain the existing video handling of invalid values."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def v6_depth_payload(result: Any) -> dict[str, float | None]:
    """Expose V6's final centimeters; never select, correct, or fuse depth."""
    return {"final_shadow_depth_cm": finite_depth(result.final_shadow_depth_cm)}
