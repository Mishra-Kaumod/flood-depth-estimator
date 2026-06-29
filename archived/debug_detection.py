#!/usr/bin/env python3
"""
Debug water detection on a single flood image to understand why it's not being detected.
"""
import cv2
import numpy as np

def debug_single_image(img_path):
    """Debug all detection methods on a single image."""
    
    print(f"\n{'='*70}")
    print(f"Debugging: {img_path}")
    print('='*70)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to read image")
        return
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    
    # Debug color analysis
    print(f"\n--- COLOR ANALYSIS ---")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    print(f"H range: {h.min()}-{h.max()}")
    print(f"S range: {s.min()}-{s.max()}")
    print(f"V range: {v.min()}-{v.max()}")
    
    # Water mask
    water_mask = cv2.inRange(hsv, (90, 0, 100), (130, 100, 255))
    saturated_mask = cv2.inRange(s, 150, 255)
    blue_hue = cv2.inRange(h, 90, 130)
    saturated_blue = cv2.bitwise_and(blue_hue, saturated_mask)
    water_mask = cv2.subtract(water_mask, saturated_blue)
    
    water_coverage = np.count_nonzero(water_mask) / water_mask.size
    print(f"Water coverage: {water_coverage:.3f} (threshold: 0.08)")
    print(f"Water detected: {'YES' if water_coverage > 0.08 else 'NO'}")
    
    # Debug edges
    print(f"\n--- EDGE DETECTION ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.count_nonzero(edges)
    print(f"Edge pixels: {edge_count}")
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
    horizontal_lines = 0
    if lines is not None:
        print(f"Total lines detected: {len(lines)}")
        for line in lines:
            rho, theta = line[0]
            if theta < 0.3 or theta > 2.8:
                horizontal_lines += 1
    
    print(f"Horizontal lines: {horizontal_lines} (threshold: 15)")
    print(f"Water detected: {'YES' if horizontal_lines > 15 else 'NO'}")
    
    # Show statistics
    print(f"\n--- PIXEL STATISTICS ---")
    blue_pixels = np.count_nonzero((h >= 90) & (h <= 130))
    print(f"Blue-hue pixels: {blue_pixels} ({100*blue_pixels/h.size:.1f}%)")
    
    high_sat = np.count_nonzero(s > 100)
    print(f"High saturation pixels: {high_sat} ({100*high_sat/s.size:.1f}%)")
    
    low_v = np.count_nonzero(v < 100)
    print(f"Dark pixels: {low_v} ({100*low_v/v.size:.1f}%)")

def main():
    flood_images = [
        "flood_dataset/train/flood/hydrated_flood_surface_0000.jpg",
        "flood_dataset/train/flood/hydrated_flood_surface_0010.jpg",
    ]
    
    for img_path in flood_images:
        debug_single_image(img_path)
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
