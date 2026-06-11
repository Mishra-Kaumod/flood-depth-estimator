#!/usr/bin/env python3
"""
Simple Water Detection Test - Core methods without complex dependencies.
"""
import os
from glob import glob
import cv2
import numpy as np

from cv_engine import FloodDepthEngine
from core_logic import TripleEnginePipeline

TEST_DIR = "test_images"


def simple_water_detection(image):
    """
    Simple water detection using basic color and edge analysis.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Blue/cyan water detection
    lower_blue = np.array([90, 20, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Dark reflective surfaces
    dark_reflective = cv2.inRange(s, 0, 100) & cv2.inRange(v, 40, 200)
    
    # Combine
    water_mask = cv2.bitwise_or(blue_mask, dark_reflective)
    
    # Water percentage
    h_img, w_img = image.shape[:2]
    water_pct = np.count_nonzero(water_mask) / (h_img * w_img)
    
    return {
        'water_detected': water_pct > 0.05,
        'water_percentage': water_pct,
        'method': 'Color-based HSV Analysis'
    }


def detect_horizontal_edges(image):
    """
    Detect horizontal edge lines typical of water surfaces.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Hough line detection - find horizontal lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    horizontal_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 20:  # Roughly horizontal
                horizontal_count += 1
    
    return {
        'horizontal_lines': int(horizontal_count),
        'water_detected': horizontal_count >= 3,
        'method': 'Horizontal Edge Detection'
    }


def main():
    print("\n" + "="*80)
    print(" WATER DETECTION ANALYSIS - Test on Flood Images")
    print("="*80)
    
    engine = FloodDepthEngine()
    ml_pipeline = TripleEnginePipeline()
    
    image_paths = sorted(glob(os.path.join(TEST_DIR, '*.jpg')))
    
    for img_path in image_paths:
        print(f"\n{'='*80}")
        print(f"Image: {os.path.basename(img_path)}")
        print(f"{'='*80}")
        
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to read")
            continue
        
        # Method 1: Classifier
        print("\n🏷️  METHOD 1: Neural Network Classifier")
        flood_prob = ml_pipeline.predict_flood_probability(img)
        print(f"   Flood Probability: {flood_prob:.1%}")
        
        # Method 2: Depth Engine
        print("\n📏 METHOD 2: Depth Estimation Engine")
        cv_results = engine.process_frame(img)
        print(f"   Strategy: {cv_results['strategy_applied']}")
        print(f"   Detected Objects: {cv_results['anchors_tracked']}")
        print(f"   Estimated Depth: {cv_results['calculated_depth_cm']}cm")
        print(f"   Confidence: {cv_results['confidence_metric']:.1%}")
        
        # Method 3: Color-based detection
        print("\n💧 METHOD 3: Color Analysis")
        color_result = simple_water_detection(img)
        print(f"   Water Detected: {'✅ YES' if color_result['water_detected'] else '❌ NO'}")
        print(f"   Water Coverage: {color_result['water_percentage']:.1%}")
        
        # Method 4: Edge detection
        print("\n➖ METHOD 4: Horizontal Edge Lines")
        edge_result = detect_horizontal_edges(img)
        print(f"   Horizontal Lines Found: {edge_result['horizontal_lines']}")
        print(f"   Water Detected: {'✅ YES' if edge_result['water_detected'] else '❌ NO'}")
        
        # Final decision
        print("\n" + "="*80)
        print("🎯 FINAL WATER DETECTION DECISION")
        print("="*80)
        
        # Voting system
        votes = 0
        if flood_prob > 0.4:
            votes += 1
        if color_result['water_detected']:
            votes += 1
        if edge_result['water_detected']:
            votes += 1
        
        final_water_detected = votes >= 2
        
        print(f"\nMethod Votes for Water: {votes}/3")
        print(f"  • Classifier: {'✅' if flood_prob > 0.4 else '❌'} ({flood_prob:.0%})")
        print(f"  • Color Analysis: {'✅' if color_result['water_detected'] else '❌'}")
        print(f"  • Edge Detection: {'✅' if edge_result['water_detected'] else '❌'}")
        
        print(f"\n{'🔴 CONSENSUS: WATER DETECTED' if final_water_detected else '🟢 CONSENSUS: NO WATER'}")
        
        if final_water_detected:
            depth = cv_results['calculated_depth_cm']
            if depth > 60:
                risk = "🔴 CRITICAL FLOODING"
            elif depth > 30:
                risk = "🟠 HIGH RISK"
            elif depth > 15:
                risk = "🟡 MODERATE RISK"
            else:
                risk = "🟢 LOW RISK"
            print(f"Risk Assessment: {risk} (Depth: {depth}cm)")
        else:
            print("Area appears safe - no water detected")
        
        print()


if __name__ == "__main__":
    main()
