#!/usr/bin/env python3
"""Test that all YOLO integration modules can be imported."""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

try:
    print("Testing imports...")
    
    from water_detection import WaterDetectionAnalyzer
    print("✓ WaterDetectionAnalyzer imported")
    
    from predict_image import SeverityPredictor
    print("✓ SeverityPredictor imported")
    
    from process_video import VideoFloodAnalyzer
    print("✓ VideoFloodAnalyzer imported")
    
    from depth_band_estimator import estimate_depth, DEPTH_BANDS
    print("✓ depth_band_estimator imported")
    
    from object_detection import ObjectDetector
    print("✓ ObjectDetector imported (YOLO)")
    
    from hybrid_depth_estimator import HybridDepthEstimator
    print("✓ HybridDepthEstimator imported")
    
    print("\n✓✓✓ ALL IMPORTS SUCCESSFUL ✓✓✓")
    print("\nModule Summary:")
    print(f"  - WaterDetectionAnalyzer: {WaterDetectionAnalyzer}")
    print(f"  - SeverityPredictor: {SeverityPredictor}")
    print(f"  - VideoFloodAnalyzer: {VideoFloodAnalyzer}")
    print(f"  - ObjectDetector: {ObjectDetector}")
    print(f"  - HybridDepthEstimator: {HybridDepthEstimator}")
    print(f"  - Depth bands: {len(DEPTH_BANDS)} classes")
    
except ImportError as e:
    print(f"✗ Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
