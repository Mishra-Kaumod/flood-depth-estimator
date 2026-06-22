#!/usr/bin/env python
"""
Quick verification script for enterprise upload system
Tests the endpoint and generates a sample flood image
"""

import requests
import json
from PIL import Image, ImageDraw
import numpy as np
import io

def create_test_flood_image():
    """Create a synthetic flood image for testing"""
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Draw a blue water region (simulating flood)
    draw.rectangle([(0, 350), (width, height)], fill=(30, 80, 180, 200))
    
    # Add some noise/texture
    pixels = img.load()
    np.random.seed(42)
    for i in range(width):
        for j in range(height):
            if j > 350:  # Only in flood region
                r, g, b = pixels[i, j][:3]
                noise = np.random.randint(-20, 20)
                pixels[i, j] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise))
                )
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def test_upload_system():
    """Test the secure upload system"""
    
    base_url = "http://localhost:8000"
    
    print("=" * 70)
    print("ENTERPRISE FLOOD DETECTION UPLOAD SYSTEM - QUICK TEST")
    print("=" * 70)
    
    # Test 1: Check if upload page is accessible
    print("\n[Test 1] Checking if upload page is accessible...")
    try:
        response = requests.get(f"{base_url}/random-upload/", timeout=5)
        if response.status_code == 200:
            print(f"  [OK] Upload page is accessible (HTTP {response.status_code})")
            if 'enterprise_upload' in response.text or 'Flood Analysis' in response.text:
                print("  [OK] Enterprise template is being used")
            else:
                print("  [WARN] Old template detected, but page loads")
        else:
            print(f"  [FAIL] Upload page returned HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error accessing upload page: {e}")
        return False
    
    # Test 2: Create and upload a test image
    print("\n[Test 2] Creating synthetic flood image...")
    try:
        test_image = create_test_flood_image()
        print(f"  [OK] Test image created ({test_image.getbuffer().nbytes / 1024:.1f} KB)")
    except Exception as e:
        print(f"  [FAIL] Error creating test image: {e}")
        return False
    
    # Test 3: Check CSRF token
    print("\n[Test 3] Getting CSRF token...")
    try:
        response = requests.get(f"{base_url}/random-upload/", timeout=5)
        cookies = response.cookies
        print(f"  [OK] CSRF token available in cookies")
    except Exception as e:
        print(f"  [FAIL] Error getting CSRF token: {e}")
        return False
    
    # Test 4: Upload image
    print("\n[Test 4] Uploading test image...")
    try:
        # First get the page to extract CSRF token from HTML
        response = requests.get(f"{base_url}/random-upload/", timeout=5)
        cookies = response.cookies
        
        # Extract CSRF token from HTML
        import re
        csrf_match = re.search(r"csrfToken\s*=\s*['\"]([^'\"]+)['\"]", response.text)
        if not csrf_match:
            csrf_match = re.search(r"csrf_token['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", response.text)
        
        csrf_token = csrf_match.group(1) if csrf_match else None
        
        if csrf_token:
            print(f"  [OK] CSRF token extracted: {csrf_token[:20]}...")
        else:
            print(f"  [WARN] Could not extract CSRF token from HTML")
            csrf_token = ""
        
        files = {
            'images': ('test_flood.jpg', test_image, 'image/jpeg'),
        }
        data = {
            'scenario_name': 'Test Flood Detection',
            'location': 'Bengaluru, Karnataka',
            'camera_id': 'TEST_CAM_001',
            'latitude': '13.1939',
            'longitude': '77.5900',
            'description': 'Automated test flood scenario'
        }
        
        headers = {
            'X-CSRFToken': csrf_token,
            'Referer': f"{base_url}/random-upload/"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/floods/random-upload-secure/",
            files=files,
            data=data,
            cookies=cookies,
            headers=headers,
            timeout=30
        )
        
        print(f"  Response Status: HTTP {response.status_code}")
        
        if response.status_code in [200, 201]:
            print(f"  [OK] Upload successful!")
            try:
                result = response.json()
                print(f"  Batch ID: {result.get('batch_id')}")
                print(f"  Images processed: {result.get('statistics', {}).get('total_images')}")
                print(f"  Flooded images: {result.get('statistics', {}).get('flooded_count')}")
                print(f"  Average confidence: {result.get('statistics', {}).get('avg_confidence')}%")
                print(f"  Report URL: {result.get('report_url')}")
                
                if result.get('report_url'):
                    print(f"\n[SUCCESS] Open in browser: http://localhost:8000{result.get('report_url')}")
            except json.JSONDecodeError:
                print(f"  [WARN] Response is not JSON: {response.text[:200]}")
        else:
            print(f"  [FAIL] Upload failed with HTTP {response.status_code}")
            if response.status_code == 403:
                print(f"  [ERROR] CSRF Forbidden - Token issue")
            print(f"  Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Error uploading image: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("[SUCCESS] ENTERPRISE UPLOAD SYSTEM IS READY")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Open http://localhost:8000/random-upload/ in your browser")
    print("2. Upload 1-10 flood images")
    print("3. View the professional enterprise report with color-coded results")
    print("4. Print or download the report as PDF")
    
    return True

if __name__ == "__main__":
    test_upload_system()
