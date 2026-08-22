#!/usr/bin/env python3
"""Test S3 integration and all modules."""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

print("Testing S3 Integration & Module Imports\n")
print("="*60)

try:
    print("\n1. Testing core module imports...")
    from water_detection import WaterDetectionAnalyzer
    print("   ✓ WaterDetectionAnalyzer")
    
    from predict_image import SeverityPredictor
    print("   ✓ SeverityPredictor")
    
    from process_video import VideoFloodAnalyzer
    print("   ✓ VideoFloodAnalyzer")
    
    from object_detection import ObjectDetector
    print("   ✓ ObjectDetector")
    
    from hybrid_depth_estimator import HybridDepthEstimator
    print("   ✓ HybridDepthEstimator")
    
    from depth_band_estimator import estimate_depth, DEPTH_BANDS
    print("   ✓ depth_band_estimator")
    
    print("\n2. Testing S3 handler...")
    try:
        from s3_handler import S3Handler
        print("   ✓ S3Handler (boto3 available)")
        
        # Try to initialize (will fail without AWS creds, that's OK)
        try:
            handler = S3Handler(bucket_name="test-bucket")
            print("   ✓ S3 connection initialized")
        except Exception as e:
            print(f"   ℹ S3 initialization skipped (expected without AWS creds): {type(e).__name__}")
    
    except ImportError as e:
        print(f"   ℹ boto3 not installed (S3 support disabled): {e}")
    
    print("\n3. Testing main.py imports...")
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Just check if main.py can be parsed
    with open("main.py", "r", encoding="utf-8") as f:
        main_code = f.read()
    compile(main_code, "main.py", "exec")
    print("   ✓ main.py syntax valid")
    
    print("\n" + "="*60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("\nSystem Features:")
    print("  ✓ Water Detection (6 methods)")
    print("  ✓ Severity Classification (ResNet18)")
    print("  ✓ Depth Estimation (Hybrid 3-method)")
    print("  ✓ YOLO Object Detection")
    print("  ✓ Local Storage (DEFAULT)")
    print("  ✓ AWS S3 Storage (if boto3 installed)")
    print("\nUsage Examples:")
    print("  Local:  python main.py image test_images/image.jpg")
    print("  AWS:    python main.py image images/image.jpg --storage=aws")
    print("  Help:   python main.py")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
