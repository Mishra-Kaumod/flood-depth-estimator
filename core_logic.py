# core_logic.py
import cv2
import numpy as np
import torch
import os
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Global Engine Caching Layer
yolo_model = YOLO("yolov8n.pt")
depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
torch.set_num_threads(2)

class TripleEnginePipeline:
    def __init__(self):
        print("[+] Loading core ensemble components into Triple-Engine Pipeline...")
        self.custom_classifier = models.mobilenet_v3_large()
        num_ftrs = self.custom_classifier.classifier[3].in_features
        self.custom_classifier.classifier[3] = torch.nn.Linear(num_ftrs, 2)
        
        weight_file = "lightweight_flood_classifier.pt"
        if os.path.exists(weight_file):
            self.custom_classifier.load_state_dict(torch.load(weight_file, map_location='cpu'))
            print("[+] Custom lightweight flood classifier weights integrated.")
        else:
            print("[!] Warning: Custom weights file not found. Running with baseline initialization.")
            
        self.custom_classifier.eval()
        self.classifier_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_flood_probability(self, cv2_image_matrix):
        """Calculates real-time confidence score for surface water presence."""
        rgb_img = cv2.cvtColor(cv2_image_matrix, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tensor_img = self.classifier_transforms(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.custom_classifier(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Index 1 tracks 'flood' probability score based on dataset alphabetic ordering
        return float(probabilities[0][1])

def estimate_flood_depth(image_array, context_profile=None):
    """Fallback standard functional loop for standalone batch testing wrappers."""
    img_height, img_width, _ = image_array.shape
    inference_size = 448
    resized_img = cv2.resize(image_array, (inference_size, inference_size), interpolation=cv2.INTER_AREA)
    
    yolo_results = yolo_model(resized_img, verbose=False)[0]
    anchors = []
    lowest_y = 0
    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        if label in ['car', 'person', 'bus', 'motorcycle', 'truck']:
            coords = box.xyxy[0].tolist()
            y2 = int(coords[3])
            anchors.append({"class": label, "bottom_edge": y2})
            if y2 > lowest_y: lowest_y = y2

    rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    inputs = depth_processor(images=rgb_image, return_tensors="pt")
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(inference_size, inference_size),
            mode="bicubic", align_corners=False
        ).squeeze()
        
    depth_map = predicted_depth.cpu().numpy()
    depth_min, depth_max = depth_map.min(), depth_map.max()
    normalized_depth = (depth_map - depth_min) / (depth_max - depth_min) if (depth_max - depth_min) > 0 else depth_map

    PROFILE_SCALERS = {
        "Hatchback/Sedan": 108.0, "SUV/Utility Truck": 128.0, "Transit Bus": 200.0,
        "Gauge Infrastructure": 180.0, "Two-Wheeler/Motorcycle": 74.5
    }
    base_scaler = PROFILE_SCALERS.get(context_profile, 165.0)

    if anchors and lowest_y > 0:
        sample_zone = normalized_depth[int(lowest_y * 0.95):lowest_y, :]
        water_plane_score = float(np.mean(sample_zone))
        estimated_depth_cm = round(water_plane_score * 120.0, 2)
        calculation_mode = f"Object Target Anchoring [{anchors[0]['class']}]"
    else:
        lower_third_start = int(inference_size * 0.70)
        water_plane_zone = normalized_depth[lower_third_start:inference_size, :]
        water_plane_score = float(np.mean(water_plane_zone))
        estimated_depth_cm = round(water_plane_score * base_scaler, 2)
        calculation_mode = f"Context-Aligned Semantic Calibration Matrix ({context_profile})"

    return {
        "status": "success",
        "calculation_mode": calculation_mode,
        "estimated_depth_cm": estimated_depth_cm,
    }
