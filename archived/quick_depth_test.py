#!/usr/bin/env python3
"""
Quick depth test - faster version without all model initialization.
"""
import os
from glob import glob
import cv2

TEST_DIR = "test_images"

def main():
    image_paths = sorted(glob(os.path.join(TEST_DIR, '*.jpg')))
    
    print("\n" + "="*70)
    print("DEPTH ESTIMATION TEST - test_images folder")
    print("="*70)
    
    if not image_paths:
        print("❌ No images found in test_images/")
        return
    
    # Import only when needed
    print("\n📊 Loading models (this may take 30-60 seconds)...\n")
    from cv_engine import FloodDepthEngine
    
    engine = FloodDepthEngine()
    
    print("\n" + "="*70)
    print("DEPTH RESULTS")
    print("="*70)
    
    results = []
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        print(f"\n[{idx}/{len(image_paths)}] {filename}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ❌ Failed to read image")
            continue
        
        try:
            metrics = engine.process_frame(img)
            depth = metrics['calculated_depth_cm']
            confidence = metrics['confidence_metric']
            anchors = metrics['anchors_tracked']
            
            print(f"  📏 Depth: {depth} cm")
            print(f"  ⭐ Confidence: {confidence:.1%}")
            print(f"  🎯 Reference Objects: {', '.join(anchors)}")
            
            results.append({
                'image': filename,
                'depth_cm': depth,
                'confidence': confidence,
                'anchors': anchors
            })
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        depths = [r['depth_cm'] for r in results]
        print(f"\nTotal Images Tested: {len(results)}")
        print(f"\nDepth Estimates:")
        for r in results:
            print(f"  • {r['image']}: {r['depth_cm']}cm")
        
        print(f"\nStatistics:")
        print(f"  Average Depth: {sum(depths) / len(depths):.2f} cm")
        print(f"  Min Depth: {min(depths):.2f} cm")
        print(f"  Max Depth: {max(depths):.2f} cm")
        
        # Risk assessment
        avg_depth = sum(depths) / len(depths)
        if avg_depth > 60:
            risk = "🔴 CRITICAL FLOODING"
        elif avg_depth > 30:
            risk = "🟠 HIGH RISK"
        elif avg_depth > 15:
            risk = "🟡 MODERATE RISK"
        else:
            risk = "🟢 LOW RISK"
        
        print(f"\n⚠️  Overall Risk Level: {risk}")
    else:
        print("❌ No results to summarize")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
