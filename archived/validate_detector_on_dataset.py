#!/usr/bin/env python3
"""
Test improved water detector on flood_dataset to verify it still detects real water.
"""
import os
from glob import glob
import cv2
import numpy as np

def test_dataset_category(detector, category_dir, expected_water, category_name, num_samples=5):
    """Test water detection on a dataset category."""
    
    print(f"\n{'='*70}")
    print(f"Testing {category_name.upper()}")
    print('='*70)
    
    image_paths = sorted(glob(os.path.join(category_dir, '*.jpg')))[:num_samples]
    
    if not image_paths:
        print(f"❌ No images found in {category_dir}")
        return 0, 0
    
    correct = 0
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  [{idx}] ❌ Failed to read {filename}")
            continue
        
        try:
            # Simple test without reference objects for these texture patches
            result = detector.detect_water_improved(img, {})
            
            # Check if result matches expectation
            matches = result['water_detected'] == expected_water
            symbol = "✅" if matches else "❌"
            expected_str = "WATER" if expected_water else "NO WATER"
            actual_str = "WATER" if result['water_detected'] else "NO WATER"
            
            print(f"  [{idx}] {symbol} {filename}")
            print(f"       Expected: {expected_str}, Got: {actual_str} (confidence: {result['confidence']:.0%})")
            
            if matches:
                correct += 1
        
        except Exception as e:
            print(f"  [{idx}] ❌ Error: {str(e)}")
    
    accuracy = correct / len(image_paths) if image_paths else 0
    print(f"\n  Result: {correct}/{len(image_paths)} correct ({accuracy:.0%})")
    
    return correct, len(image_paths)

def main():
    print("\n" + "="*70)
    print("VALIDATING IMPROVED WATER DETECTOR ON FLOOD DATASET")
    print("="*70)
    
    from improved_water_detector import ImprovedWaterDetector
    
    detector = ImprovedWaterDetector()
    
    # Test on flood_dataset
    dry_dir = "flood_dataset/train/dry"
    flood_dir = "flood_dataset/train/flood"
    
    dry_correct, dry_total = test_dataset_category(
        detector, dry_dir, False, "Dry Images", num_samples=5
    )
    
    flood_correct, flood_total = test_dataset_category(
        detector, flood_dir, True, "Flood Images", num_samples=5
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print('='*70)
    
    total_correct = dry_correct + flood_correct
    total_images = dry_total + flood_total
    
    if total_images > 0:
        overall_accuracy = total_correct / total_images
        print(f"\n✅ Overall Accuracy: {total_correct}/{total_images} ({overall_accuracy:.0%})")
        
        if overall_accuracy >= 0.7:
            print(f"🟢 GOOD: Detector is working reasonably well")
        elif overall_accuracy >= 0.5:
            print(f"🟡 FAIR: Detector needs improvement")
        else:
            print(f"🔴 POOR: Detector needs significant rework")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
