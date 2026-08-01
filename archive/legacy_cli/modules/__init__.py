"""Public module exports for the cleaned flood analysis project."""

try:
    from .water_detection import WaterDetectionAnalyzer
except Exception:  # pragma: no cover - optional runtime dependency
    WaterDetectionAnalyzer = None

try:
    from .predict_image import SeverityPredictor
except Exception:  # pragma: no cover - optional runtime dependency
    SeverityPredictor = None

try:
    from .process_video import VideoFloodAnalyzer
except Exception:  # pragma: no cover - optional runtime dependency
    VideoFloodAnalyzer = None

try:
    from .depth_band_estimator import estimate_depth, DEPTH_BANDS
except Exception:  # pragma: no cover - optional runtime dependency
    estimate_depth = None
    DEPTH_BANDS = {}

try:
    from .object_detection import ObjectDetector
except Exception:  # pragma: no cover - optional runtime dependency
    ObjectDetector = None

try:
    from .hybrid_depth_estimator import HybridDepthEstimator
except Exception:  # pragma: no cover - optional runtime dependency
    HybridDepthEstimator = None

try:
    from .flood_analyzer import FloodAnalyzer
except Exception:  # pragma: no cover - optional runtime dependency
    FloodAnalyzer = None

try:
    from .production_pipeline import ProductionFloodAnalyzer
except Exception:  # pragma: no cover - optional runtime dependency
    ProductionFloodAnalyzer = None

try:
    from .s3_handler import S3Handler
    S3_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    S3_AVAILABLE = False
    S3Handler = None

__version__ = "2.1.0"
__all__ = [
    "WaterDetectionAnalyzer",
    "SeverityPredictor",
    "VideoFloodAnalyzer",
    "ObjectDetector",
    "HybridDepthEstimator",
    "FloodAnalyzer",
    "ProductionFloodAnalyzer",
    "S3Handler",
    "estimate_depth",
    "DEPTH_BANDS",
    "S3_AVAILABLE",
]
