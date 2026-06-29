#!/usr/bin/env python3
"""
Test improved water detection on test_images to verify hallucination fix.
"""
import os
from glob import glob
import cv2
import sys

TEST_DIR = "test_images"

def main():
    print("\n" + "="*70)
    print("TESTING IMPROVED WATER DETECTION - test_images")
    print("="*70 + "\n")
    
    # Import after print for cleaner output
    from improved_water_detector import ImprovedWaterDetector
    from cv_engine import FloodDepthEngine
    
    detector = ImprovedWaterDetector()
    engine = FloodDepthEngine()
    
    image_paths = sorted(glob(os.path.join(TEST_DIR, '*.jpg')))
    
    if not image_paths:
        print("❌ No images found in test_images/")
        return
    
    print(f"📊 Testing {len(image_paths)} images...\n")
    
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        print(f"\n{'='*70}")
        print(f"Image {idx}/{len(image_paths)}: {filename}")
        print('='*70)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to read image\n")
            continue
        
        try:
            # Get depth metrics (includes detected objects)
            metrics = engine.process_frame(img)
            
            # Prepare detected objects dict
            detected_objects = {}
            if 'anchors_tracked' in metrics:
                anchors = metrics['anchors_tracked']
                # Map anchor names to dict keys
                if 'person' in anchors:
                    detected_objects['persons'] = [(10, 10, 100, 200)]  # Placeholder bbox
                if 'bus' in anchors:
                    detected_objects['buses'] = [(150, 20, 400, 350)]  # Placeholder bbox
                if 'car' in anchors:
                    detected_objects['cars'] = [(10, 10, 150, 100)]
            
            # Run improved water detection
            result = detector.detect_water_improved(img, detected_objects)
            
            # Display results
            print(f"\n📊 DETECTION RESULTS:")
            print(f"  Water Detected: {'🔴 YES' if result['water_detected'] else '🟢 NO'}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  Method: {result.get('method', 'consensus')}")
            print(f"  Is Hallucination: {'❌ YES' if result.get('is_hallucination') else '✅ NO'}")
            
            print(f"\n💾 REFERENCE OBJECTS DETECTED:")
            if 'anchors_tracked' in metrics:
                for anchor in metrics['anchors_tracked']:
                    print(f"  ✓ {anchor}")
            else:
                print(f"  (none)")
            
            print(f"\n📏 DEPTH ESTIMATE:")
            print(f"  Depth: {metrics['calculated_depth_cm']} cm")
            print(f"  Confidence: {metrics['confidence_metric']:.1%}")
            
            print(f"\n📋 DETECTION REASONS:")
            for reason in result['reasons']:
                print(f"  • {reason}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
