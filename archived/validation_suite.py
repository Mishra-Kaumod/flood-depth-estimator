import os
import cv2

# Define Ground Truth: Scenario Name -> Expected Status (True=Flooded, False=Dry)
ground_truth = {
    "deep_flood": True, "shallow_flood": True, "muddy_flood": True,
    "night_flood": True, "rushing_water": True, "debris_flood": True,
    "submerged_car": True, "submerged_bus": True, "curb_flood": True, "storm_surge": True,
    "dry_street": False, "dry_night": False, "blue_bus": False, 
    "small_puddles": False, "wet_asphalt": False, "shadows": False,
    "traffic_jam": False, "blue_tarp": False, "raindrops": False, "snow": False
}

def validate():
    print(f"\n{'='*50}\nRUNNING VALIDATION REPORT\n{'='*50}")
    
    # Run the existing ensemble logic (Imported)
    from terminal_test import cv2_ensemble_estimator
    
    # Load Dry Reference (Required for the ensemble)
    dry_ref = cv2.imread("reference_images/dry_cam_02.jpg")
    
    test_folder = "test_images"
    files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png'))]
    
    passed = 0
    total = len(files)
    
    for f in files:
        # Determine reality from filename
        is_flooded_reality = any(key in f for key in ground_truth if ground_truth[key] == True)
        
        # Get System result
        img = cv2.imread(os.path.join(test_folder, f))
        depth, status = cv2_ensemble_estimator(img, dry_ref)
        is_flooded_system = (depth > 0.0)
        
        # Compare
        result = "PASS" if (is_flooded_reality == is_flooded_system) else "FAIL"
        if result == "PASS": passed += 1
        
        print(f"File: {f[:20]}... | Truth: {'Flood' if is_flooded_reality else 'Dry'} | System: {status} ({depth}cm) | {result}")

    print(f"\nFINAL ACCURACY: {round((passed/total)*100, 1)}%")

if __name__ == "__main__":
    validate()
