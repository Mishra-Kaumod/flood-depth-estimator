#!/usr/bin/env python3
"""
EXAMPLE: Flood Depth Estimation - Temporal Analysis Testing

This script demonstrates how to:
1. Upload multiple images from the same camera over time
2. Trigger temporal analysis
3. Retrieve consensus depth estimates
4. Check hallucination prevention

Usage:
    python example_temporal_usage.py

Requirements:
    - Django server running (localhost:8000)
    - Test images in ./test_images/ directory
    - requests library: pip install requests
"""

import requests
import time
import os
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
CAMERA_ID = "demo_camera_01"
LOCATION_NAME = "Demo Location - Main Street"

# API Endpoints
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/estimate/"
TEMPORAL_ENDPOINT = f"{API_BASE_URL}/api/v1/temporal/{CAMERA_ID}/"
TEMPORAL_ANALYZE_ENDPOINT = f"{API_BASE_URL}/api/v1/temporal/{CAMERA_ID}/analyze/"
STATS_ENDPOINT = f"{API_BASE_URL}/api/v1/camera/{CAMERA_ID}/stats/"

def upload_image(image_path, context=""):
    """
    Upload a single image to the system.
    
    Args:
        image_path: Path to image file
        context: Additional context about the situation
        
    Returns:
        Response JSON
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    with open(image_path, 'rb') as img_file:
        files = {
            'image': img_file,
        }
        data = {
            'camera_id': CAMERA_ID,
            'location_name': LOCATION_NAME,
            'context': context
        }
        
        try:
            response = requests.post(UPLOAD_ENDPOINT, files=files, data=data)
            return response.json()
        except Exception as e:
            print(f"❌ Error uploading {image_path}: {e}")
            return None


def simulate_camera_sequence(test_images_dir="./test_images", num_images=5, interval_seconds=120):
    """
    Simulate a camera sending multiple images over time.
    
    This demonstrates the 5-15 minute interval requirement:
    - 5 images × 2 minutes = 10 minutes total
    - Should trigger temporal analysis when 5th image arrives
    
    Args:
        test_images_dir: Directory containing test images
        num_images: Number of images to upload
        interval_seconds: Time between uploads (seconds)
    """
    
    print(f"\n{'='*70}")
    print(f"📷 SIMULATING CAMERA SEQUENCE")
    print(f"{'='*70}")
    print(f"Camera ID: {CAMERA_ID}")
    print(f"Location: {LOCATION_NAME}")
    print(f"Uploading {num_images} images at {interval_seconds}s intervals")
    print(f"Total time: {num_images * interval_seconds / 60:.1f} minutes")
    print()
    
    # Get test images
    test_images = list(Path(test_images_dir).glob("*.jpg"))[:num_images]
    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        print("   Please add some .jpg files to test_images/ directory")
        return
    
    test_images = sorted(test_images)  # Ensure consistent order
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n[{i}/{num_images}] Uploading: {image_path.name}")
        
        context = f"Heavy rainfall, stream nearby (Frame {i}/{num_images})"
        response = upload_image(str(image_path), context)
        
        if response:
            print(f"   Status: {response.get('status')}")
            print(f"   Queue: {response.get('queue_percentage', 0):.0f}%")
            if response.get('message'):
                print(f"   Message: {response['message']}")
            
            # When we hit 5 images, temporal analysis triggers
            if response.get('status') == 'processing':
                print(f"   ✅ TEMPORAL ANALYSIS TRIGGERED!")
                print(f"   Waiting 30 seconds for analysis to complete...")
                time.sleep(30)
        
        # Wait before next upload (simulating real-time camera feed)
        if i < num_images:
            print(f"   ⏳ Waiting {interval_seconds}s before next frame...")
            time.sleep(interval_seconds)


def get_temporal_sequence():
    """
    Retrieve the most recent temporal sequence for the camera.
    """
    print(f"\n{'='*70}")
    print(f"📊 RETRIEVING TEMPORAL SEQUENCE")
    print(f"{'='*70}")
    
    try:
        response = requests.get(TEMPORAL_ENDPOINT)
        result = response.json()
        
        if result.get('status') == 'success':
            print(f"✅ Sequence Found (ID: {result['sequence_id']})")
            print()
            print(f"   Images Analyzed: {result['num_images']}")
            print(f"   Time Span: {result['time_span_minutes']:.1f} minutes")
            print(f"   Reference Objects: {', '.join(result['detected_anchor_types'])}")
            print()
            print(f"   📏 DEPTH ESTIMATES:")
            print(f"      Average: {result['average_depth_cm']}cm")
            print(f"      Min: {result['min_depth_cm']}cm")
            print(f"      Max: {result['max_depth_cm']}cm")
            print()
            print(f"   💧 WATER VALIDATION:")
            print(f"      Consensus: {'✅ YES' if result['consensus_water_present'] else '❌ NO'}")
            print(f"      Confidence: {result['confidence_score']:.1%}")
            print()
            
            # Risk assessment
            if result['average_depth_cm'] is not None:
                depth = result['average_depth_cm']
                if depth < 15:
                    risk = "🟢 LOW"
                elif depth < 30:
                    risk = "🟡 MODERATE"
                elif depth < 60:
                    risk = "🟠 HIGH"
                else:
                    risk = "🔴 CRITICAL"
                print(f"   ⚠️  RISK LEVEL: {risk} (Depth: {depth}cm)")
            
            return result
        else:
            print(f"❌ {result.get('message', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error retrieving sequence: {e}")
        return None


def get_camera_stats(hours=1):
    """
    Get statistics for the camera over the past N hours.
    """
    print(f"\n{'='*70}")
    print(f"📈 CAMERA STATISTICS (Last {hours} hours)")
    print(f"{'='*70}")
    
    try:
        response = requests.get(STATS_ENDPOINT, params={'hours': hours})
        stats = response.json()
        
        if stats.get('status') == 'success':
            print(f"✅ Camera: {stats['camera_name']}")
            print()
            print(f"   Total Images: {stats['total_images']}")
            print(f"   Water-Confirmed: {stats['water_confirmed_images']}")
            if stats['total_images'] > 0:
                pct = (stats['water_confirmed_images'] / stats['total_images']) * 100
                print(f"   Water Confirmation Rate: {pct:.1f}%")
            print()
            print(f"   Depth Statistics:")
            print(f"      Average: {stats['avg_depth_cm']}cm")
            print(f"      Maximum: {stats['max_depth_cm']}cm")
            print()
            print(f"   Temporal Sequences: {stats['temporal_sequences']}")
            
            return stats
        else:
            print(f"❌ {stats.get('message', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error retrieving stats: {e}")
        return None


def demonstrate_hallucination_prevention():
    """
    Show examples of hallucination prevention in action.
    """
    print(f"\n{'='*70}")
    print(f"🛡️  HALLUCINATION PREVENTION EXAMPLES")
    print(f"{'='*70}")
    
    print("""
The system prevents false positives through multiple mechanisms:

1️⃣  CASE: No Reference Objects Detected
   Image has depth estimate but no person/car/bus/truck/motorcycle
   ❌ REJECTED: "No reference objects - cannot validate depth"
   
2️⃣  CASE: Single Reference Object + Low Water Probability  
   Only 1 car detected, water confidence = 35%
   ⚠️  LOW CONFIDENCE: "Only 1 reference object - need sequence validation"
   ✅ ACCEPTED only if 5+ images confirm (multi-image consensus)
   
3️⃣  CASE: Multiple Objects BUT No Water Detected
   5 images, 3 reference objects, but water probability avg = 20%
   ❌ REJECTED: "No water consensus (20% < 40% threshold)"
   
4️⃣  CASE: Clear Flooding (NO False Positive)
   10 images, 4 reference object types
   Water probability avg = 72% across all images
   Depth readings: 54, 56, 55, 57, 56cm (consistent)
   ✅ VALIDATED: "High confidence - multiple anchors confirm water"

KEY METRICS:
   - 1 anchor type: Need 5+ images to validate
   - 2 anchor types: Need 3+ images to validate  
   - 3+ anchor types: Immediately validated
   - Water probability: Must average > 40% for confirmation
   - Depth consistency: Must have low std deviation (< 5cm)
    """)


def example_workflow():
    """
    Complete example workflow demonstrating the enhanced system.
    """
    print("\n" + "="*70)
    print("🌊 FLOOD DEPTH ESTIMATION - TEMPORAL ANALYSIS DEMO")
    print("="*70)
    
    # Step 1: Upload images
    print("\n📍 STEP 1: Uploading multiple images from same camera...")
    simulate_camera_sequence(
        test_images_dir="./test_images",
        num_images=5,
        interval_seconds=2  # 2 seconds for demo (normally 120 seconds = 2 minutes)
    )
    
    # Step 2: Retrieve temporal sequence
    print("\n📍 STEP 2: Retrieving temporal sequence analysis...")
    time.sleep(5)  # Give the system time to process
    sequence = get_temporal_sequence()
    
    # Step 3: Get camera statistics
    if sequence:
        print("\n📍 STEP 3: Retrieving camera statistics...")
        get_camera_stats(hours=1)
    
    # Step 4: Show hallucination prevention
    print("\n📍 STEP 4: Understanding hallucination prevention...")
    demonstrate_hallucination_prevention()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)


# ============================================================================
# QUICK REFERENCE: MANUAL API CALLS
# ============================================================================

QUICK_REFERENCE = """
🔧 QUICK API REFERENCE

1️⃣  UPLOAD IMAGE:
    POST /api/v1/estimate/
    {
        'image': <file>,
        'camera_id': 'camera_01',
        'location_name': 'Main Street',
        'latitude': 40.7128,
        'longitude': -74.0060,
        'context': 'Heavy rainfall'
    }

2️⃣  GET TEMPORAL SEQUENCE:
    GET /api/v1/temporal/camera_01/
    Response includes:
    - num_images: 5
    - average_depth_cm: 42.5
    - detected_anchor_types: ["car", "person"]
    - consensus_water_present: true
    - confidence_score: 0.856

3️⃣  TRIGGER TEMPORAL ANALYSIS:
    POST /api/v1/temporal/camera_01/analyze/

4️⃣  GET CAMERA STATS:
    GET /api/v1/camera/camera_01/stats/?hours=24
    Response includes:
    - total_images: 47
    - water_confirmed_images: 18
    - avg_depth_cm: 38.5
    - temporal_sequences: 6

📊 KEY THRESHOLDS:
   - Buffer trigger: 5 images
   - Time window: 5-15 minutes
   - Min water confidence: 40%
   - Min anchors: 1 (with validation), ideally 2+
   - Depth consistency std dev: < 5cm for validation
"""

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "upload":
            image_path = sys.argv[2] if len(sys.argv) > 2 else "./test_images/test_1.jpg"
            upload_image(image_path, "Demo upload")
            
        elif command == "sequence":
            get_temporal_sequence()
            
        elif command == "stats":
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            get_camera_stats(hours)
            
        elif command == "prevent":
            demonstrate_hallucination_prevention()
            
        elif command == "full":
            example_workflow()
            
        elif command == "reference":
            print(QUICK_REFERENCE)
        else:
            print(f"Unknown command: {command}")
            print("Available: upload, sequence, stats, prevent, full, reference")
    else:
        # Run full demo if no arguments
        example_workflow()
        print(QUICK_REFERENCE)
