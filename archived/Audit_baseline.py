import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import pandas as pd

# 1. Initialize DeepLabV3
print("Loading Model...")
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_iou(pred_mask, true_mask):
    """Calculates Intersection over Union for Semantic Masks"""
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    if np.sum(union) == 0: return 1.0 
    return np.sum(intersection) / np.sum(union)

# 2. Set Paths for the Kaggle Dataset
BASE_DIR = "dataset_labeled/"
IMAGES_DIR = os.path.join(BASE_DIR, "Image")
MASKS_DIR = os.path.join(BASE_DIR, "Mask")

print("Starting Baseline Audit...")
iou_scores = []

# 3. Load Metadata
try:
    metadata = pd.read_csv(os.path.join(BASE_DIR, "metadata.csv"))
except FileNotFoundError:
    print("Error: metadata.csv not found.")
    exit()

# 4. Execution Loop
for index, row in metadata.iterrows():
    img_name = row['Image']
    mask_name = row['Mask']
    
    img_path = os.path.join(IMAGES_DIR, img_name)
    mask_path = os.path.join(MASKS_DIR, mask_name)
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue 
        
    # Load Image and Ground Truth Mask
    img = cv2.imread(img_path)
    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # --- ROBUSTNESS & ACCURACY FIXES ---
    
    # 1. Prevent NoneType Crash (Catches corrupted Kaggle downloads)
    if img is None or true_mask is None:
        print(f"Warning: OpenCV could not read bytes for {img_name}. Skipping.")
        continue
        
    # 2. Accuracy Fix: OpenCV reads BGR, but DeepLabV3 expects RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # -----------------------------------

    # Standardize ground truth to boolean (0 or 1)
    true_mask = (true_mask > 127).astype(np.uint8)
    
    # Run AI Prediction
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Extract Class 15 (Road)
    pred_mask = output.argmax(0).byte().cpu().numpy()
    pred_road_mask = (pred_mask == 15).astype(np.uint8)
    
    # Resize prediction to match original image dimensions
    pred_road_mask = cv2.resize(pred_road_mask, (true_mask.shape[1], true_mask.shape[0]))
    
    # Calculate Performance
    iou = calculate_iou(pred_road_mask, true_mask)
    iou_scores.append(iou)
    print(f"Processed {img_name} | IoU Score: {iou:.4f}")

# 5. Final Executive Output
if iou_scores:
    mean_iou = np.mean(iou_scores)
    print("-" * 40)
    print(f"BASELINE SYSTEM IoU: {mean_iou:.4f} ({(mean_iou*100):.1f}%)")
    print("Target for Production: > 0.8500")
    print("-" * 40)
else:
    print("No images were successfully processed.")