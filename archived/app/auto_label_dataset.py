import os
import json
import glob
import random

# 1. Configuration (Pointed directly at your extracted folder)
IMAGE_DIR = "/workspaces/flood-depth-estimator/test_images/team"
OUTPUT_FILE = "ground_truth_labels.json"

def analyze_image(image_path, filename):
    """Bypasses the OpenAI API and generates a mathematically sound JSON payload for testing."""
    
    # Simulate a realistic distribution of flood severities
    mock_severity = random.choices(
        [0, 1, 2, 3, 4], 
        weights=[0.4, 0.2, 0.2, 0.1, 0.1]
    )[0]
    
    # Map the severity to the depth bands and generate a randomized physical depth
    severity_map = {
        0: {"name": "Low", "band": "0-5 cm", "cm": random.randint(0, 5)},
        1: {"name": "Minor", "band": "5-20 cm", "cm": random.randint(6, 20)},
        2: {"name": "Moderate", "band": "20-50 cm", "cm": random.randint(21, 50)},
        3: {"name": "High", "band": "50-80 cm", "cm": random.randint(51, 80)},
        4: {"name": "Severe", "band": "80+ cm", "cm": random.randint(81, 120)}
    }
    
    return {
        "image_filename": filename,
        "severity": mock_severity,
        "severity_name": severity_map[mock_severity]["name"],
        "estimated_depth_band": severity_map[mock_severity]["band"],
        "actual_depth_cm": severity_map[mock_severity]["cm"],
        "objects": random.sample(["car", "pedestrian", "truck", "bus"], k=random.randint(0,2)),
        "confidence": round(random.uniform(0.75, 0.99), 3)
    }

if __name__ == "__main__":
    print("Starting Local Offline Ground-Truth Generation...")
    
    # Grab all JPGs in the directory
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}. Check your path.")
        exit()

    print(f"Found {len(image_paths)} images. Processing offline...")

    master_dataset = []
    
    # Processing Loop
    for idx, path in enumerate(image_paths):
        filename = os.path.basename(path)
        
        # Generate the mock payload
        payload = analyze_image(path, filename)
        master_dataset.append(payload)
        
        # Print a status update every 50 images so you know it hasn't frozen
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images...")

    # Save the final array to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(master_dataset, f, indent=2)

    print(f"\n✅ Generation Complete! All mock payloads successfully saved to '{OUTPUT_FILE}'.")