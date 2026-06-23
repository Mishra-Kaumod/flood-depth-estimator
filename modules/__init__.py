"""
Flood Detection & Depth Estimation Modules

Core modules:
- water_detection: Advanced water surface detection
- predict_image: Flood severity classification
- process_video: Video processing pipeline
- depth_band_estimator: Map severity to depth
- object_detection: YOLO-based object detection
- hybrid_depth_estimator: Multi-method depth estimation with YOLO
- s3_handler: AWS S3 file operations
"""

from .water_detection import WaterDetectionAnalyzer
from .predict_image import SeverityPredictor
from .process_video import VideoFloodAnalyzer
from .depth_band_estimator import estimate_depth, DEPTH_BANDS
from .object_detection import ObjectDetector
from .hybrid_depth_estimator import HybridDepthEstimator
from .flood_analyzer import FloodAnalyzer

try:
    from .s3_handler import S3Handler
    S3_AVAILABLE = True
except ImportError:
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
    "S3Handler",
    "estimate_depth",
    "DEPTH_BANDS",
    "S3_AVAILABLE"
]
