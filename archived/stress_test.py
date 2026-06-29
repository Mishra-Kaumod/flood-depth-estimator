import os
import cv2
from terminal_test import cv2_ensemble_estimator

def run_stress_test(dataset_path="master_dataset"):
    # Load your dry reference image
    dry_ref = cv2.imread("reference_images/dry_cam_02.jpg")
    
    # Track performance
    results = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    print(f"{'IMAGE NAME':<30} | {'TRUTH':<10} | {'SYSTEM':<10} | {'RESULT'}")
    print("-" * 75)
    
    for category in ["floods", "dry"]:
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder): continue
        
        for img_name in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, img_name))
            if img is None: continue
            
            # Get model inference
            depth, status = cv2_ensemble_estimator(img, dry_ref, debug=True)
            is_flooded_system = (depth > 0.0)
            is_flooded_truth = (category == "floods")
            
            # Confusion Matrix Logic
            if is_flooded_truth and is_flooded_system: results["TP"] += 1
            elif not is_flooded_truth and not is_flooded_system: results["TN"] += 1
            elif not is_flooded_truth and is_flooded_system: results["FP"] += 1
            elif is_flooded_truth and not is_flooded_system: results["FN"] += 1
            
            result = "PASS" if (is_flooded_system == is_flooded_truth) else "FAIL"
            print(f"{img_name[:30]:<30} | {'Flood' if is_flooded_truth else 'Dry':<10} | {status:<10} | {result}")

    # Summary
    print("-" * 75)
    print(f"TP:{results['TP']} | TN:{results['TN']} | FP:{results['FP']} | FN:{results['FN']}")
    print(f"Accuracy: {((results['TP']+results['TN'])/100)*100}%")

if __name__ == "__main__":
    run_stress_test()
