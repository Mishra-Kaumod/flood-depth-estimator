#!/usr/bin/env python3
"""Test that all YOLO integration modules can be imported."""

import sys

try:
    print("Testing imports...")

    from archive.legacy_cli.modules.water_detection import WaterDetectionAnalyzer
    print("[OK] WaterDetectionAnalyzer imported")

    from archive.legacy_cli.modules.predict_image import SeverityPredictor
    print("[OK] SeverityPredictor imported")

    from archive.legacy_cli.modules.process_video import VideoFloodAnalyzer
    print("[OK] VideoFloodAnalyzer imported")

    from archive.legacy_cli.modules.depth_band_estimator import estimate_depth, DEPTH_BANDS
    print("[OK] depth_band_estimator imported")

    from archive.legacy_cli.modules.object_detection import ObjectDetector
    print("[OK] ObjectDetector imported (YOLO)")

    from archive.legacy_cli.modules.hybrid_depth_estimator import HybridDepthEstimator
    print("[OK] HybridDepthEstimator imported")

    print("\n[OK][OK][OK] ALL IMPORTS SUCCESSFUL [OK][OK][OK]")
    print("\nModule Summary:")
    print(f"  - WaterDetectionAnalyzer: {WaterDetectionAnalyzer}")
    print(f"  - SeverityPredictor: {SeverityPredictor}")
    print(f"  - VideoFloodAnalyzer: {VideoFloodAnalyzer}")
    print(f"  - ObjectDetector: {ObjectDetector}")
    print(f"  - HybridDepthEstimator: {HybridDepthEstimator}")
    print(f"  - Depth bands: {len(DEPTH_BANDS)} classes")

except ImportError as e:
    print(f"[ERROR] Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error: {e}")
    sys.exit(1)
