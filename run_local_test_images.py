#!/usr/bin/env python3
"""
Run FloodDepthEngine on images in test_images/ and print results.
"""
import os
from glob import glob
import cv2
from cv_engine import FloodDepthEngine

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_images')


def main():
    engine = FloodDepthEngine()
    image_paths = sorted(glob(os.path.join(TEST_DIR, '*.jpg')))
    if not image_paths:
        print('No test images found in', TEST_DIR)
        return

    for p in image_paths:
        print('\n---')
        print('Processing:', os.path.basename(p))
        img = cv2.imread(p)
        if img is None:
            print('  Failed to read image')
            continue
        metrics = engine.process_frame(img)
        print('  Strategy:', metrics.get('strategy_applied'))
        print('  Anchors:', metrics.get('anchors_tracked'))
        print('  Num anchors:', metrics.get('num_anchors_detected'))
        print('  Depth (cm):', metrics.get('calculated_depth_cm'))
        print('  Confidence:', metrics.get('confidence_metric'))
        print('  Fallback mode:', metrics.get('is_fallback_mode'))


if __name__ == '__main__':
    main()
