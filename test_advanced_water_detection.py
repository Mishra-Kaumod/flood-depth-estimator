#!/usr/bin/env python3
"""
Test advanced water detection on test images.
Compares traditional flood classifier with new multi-method detection.
"""
import os
from glob import glob
import cv2
import numpy as np
from pathlib import Path

from water_detection import WaterDetectionAnalyzer
from cv_engine import FloodDepthEngine
from core_logic import TripleEnginePipeline

TEST_DIR = "test_images"

def main():
    print("\n" + "="*80)
    print(" ADVANCED WATER DETECTION TEST - Comprehensive Analysis")
    print("="*80)
    
    # Initialize detectors
    analyzer = WaterDetectionAnalyzer()
    engine = FloodDepthEngine()
    ml_pipeline = TripleEnginePipeline()
    
    image_paths = sorted(glob(os.path.join(TEST_DIR, '*.jpg')))
    
    for img_path in image_paths:
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(img_path)}")
        print(f"{'='*80}")
        
        img = cv2.imread(img_path)
        if img is None:
            print("  ❌ Failed to read image")
            continue
        
        # Get depth map for water detection
        h, w = img.shape[:2]
        inference_size = 448
        resized_img = cv2.resize(img, (inference_size, inference_size))
        
        # Use Depth Anything V2
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch
        
        depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        inputs = depth_processor(images=rgb_img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze()
        
        depth_array = predicted_depth.cpu().numpy()
        depth_min, depth_max = depth_array.min(), depth_array.max()
        depth_normalized = (depth_array - depth_min) / (depth_max - depth_min) if (depth_max - depth_min) > 0 else depth_array
        
        # ====================================================================
        # 1. TRADITIONAL FLOOD DETECTION (Classifier)
        # ====================================================================
        print("\n📊 METHOD 1: Traditional Classifier")
        print("-" * 80)
        flood_prob = ml_pipeline.predict_flood_probability(img)
        print(f"  Flood Probability: {flood_prob:.2%}")
        print(f"  Status: {'✅ Water Detected' if flood_prob > 0.5 else '❌ No Water'}")
        
        # ====================================================================
        # 2. CV ENGINE DEPTH ESTIMATION
        # ====================================================================
        print("\n📏 METHOD 2: Traditional Depth Estimation")
        print("-" * 80)
        cv_results = engine.process_frame(img)
        print(f"  Strategy: {cv_results['strategy_applied']}")
        print(f"  Detected Objects: {cv_results['anchors_tracked']}")
        print(f"  Estimated Depth: {cv_results['calculated_depth_cm']}cm")
        print(f"  Confidence: {cv_results['confidence_metric']:.2%}")
        
        # ====================================================================
        # 3. ADVANCED MULTI-METHOD WATER DETECTION
        # ====================================================================
        print("\n💧 METHOD 3: Advanced Multi-Method Water Detection")
        print("-" * 80)
        detection_result = analyzer.detect_water_surface(img, depth_normalized)
        
        # Print detailed report
        report = analyzer.generate_report(detection_result)
        print(report)
        
        # Print individual method results
        print("\n  Detailed Method Results:")
        print("  " + "-" * 76)
        for method_name, method_result in detection_result['details'].items():
            if method_result is None:
                continue
            status = "✅" if method_result.get('water_detected') else "❌"
            print(f"    {status} {method_name}")
            
            # Print method-specific metrics
            if 'percentage' in method_result:
                print(f"       └─ Coverage: {method_result['percentage']:.1%}")
            if 'horizontal_lines' in method_result:
                print(f"       └─ Horizontal Lines: {method_result['horizontal_lines']}")
            if 'edge_strength' in method_result:
                print(f"       └─ Edge Strength: {method_result['edge_strength']:.3f}")
            if 'avg_contrast' in method_result:
                print(f"       └─ Avg Contrast: {method_result['avg_contrast']:.3f}")
            if 'discontinuity_percentage' in method_result:
                print(f"       └─ Discontinuity: {method_result['discontinuity_percentage']:.1%}")
            if 'ripple_percentage' in method_result:
                print(f"       └─ Ripple Coverage: {method_result['ripple_percentage']:.1%}")
        
        # ====================================================================
        # 4. CONSENSUS DECISION
        # ====================================================================
        print("\n🎯 FINAL DECISION")
        print("-" * 80)
        
        # Combine classifier + multi-method
        combined_water_confidence = (flood_prob + detection_result['confidence']) / 2
        
        final_decision = detection_result['water_detected'] and flood_prob > 0.3
        
        print(f"  Classifier Confidence: {flood_prob:.1%}")
        print(f"  Multi-Method Confidence: {detection_result['confidence']:.1%}")
        print(f"  Combined Confidence: {combined_water_confidence:.1%}")
        print()
        print(f"  ⚠️  FINAL RESULT: {'✅ WATER CONFIRMED' if final_decision else '❌ NO WATER DETECTED'}")
        print()
        
        if final_decision:
            if cv_results['calculated_depth_cm'] > 60:
                risk = "🔴 CRITICAL"
            elif cv_results['calculated_depth_cm'] > 30:
                risk = "🟠 HIGH"
            elif cv_results['calculated_depth_cm'] > 15:
                risk = "🟡 MODERATE"
            else:
                risk = "🟢 LOW"
            print(f"  Risk Level: {risk} (Depth: {cv_results['calculated_depth_cm']}cm)")
        else:
            print(f"  ✅ Area is safe - no flooding detected")
        
        print()

if __name__ == "__main__":
    main()
