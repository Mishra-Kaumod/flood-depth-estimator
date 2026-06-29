#!/usr/bin/env python
"""Test the image analysis function directly"""

import sys
import json
from PIL import Image, ImageDraw
import numpy as np

# Add the flood_api module to path
sys.path.insert(0, 'C:\\Users\\pooja\\copilot-worktrees\\flood-depth-estimator\\kaumod-automatic-barnacle')

# Import the analysis function
from flood_api.secure_random_image_views import analyze_image_secure

def create_test_image(filename):
    """Create a synthetic flood image"""
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Draw a blue water region
    draw.rectangle([(0, 350), (width, height)], fill=(30, 80, 180, 200))
    
    # Add noise
    pixels = img.load()
    for i in range(width):
        for j in range(height):
            if j > 350:
                r, g, b = pixels[i, j][:3]
                noise = np.random.randint(-20, 20)
                pixels[i, j] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise))
                )
    
    img.save(filename, 'JPEG')
    return filename

print("[INFO] Creating test image...")
test_file = create_test_image('test_image_temp.jpg')
print(f"[OK] Created {test_file}")

print("\n[INFO] Analyzing image...")
result, error = analyze_image_secure(test_file)

if error:
    print(f"[ERROR] Analysis failed: {error}")
else:
    print("[OK] Analysis successful!")
    print("\n[DEBUG] Result types:")
    for key, value in result.items():
        print(f"  {key}: {type(value).__name__} = {value}")
    
    print("\n[DEBUG] Attempting JSON serialization...")
    try:
        json_str = json.dumps({'result': result})
        print("[OK] JSON serialization successful!")
        print(f"JSON: {json_str}")
    except Exception as e:
        print(f"[ERROR] JSON serialization failed: {e}")

import os
os.remove(test_file)
print("\n[INFO] Cleaned up test file")
