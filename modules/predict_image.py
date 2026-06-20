"""
FLOOD SEVERITY IMAGE PREDICTOR

Classifies flood severity from a single image using ResNet18.
Outputs: Severity class (0-4), confidence, and estimated depth.
"""

import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


class SeverityPredictor:
    """
    Predict flood severity from image.
    """
    
    def __init__(self, model_path="severity_model.pth", device=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)
        
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Severity labels
        self.severity_names = {
            0: "No / Very Low Flood",
            1: "Minor Flood",
            2: "Moderate Flood",
            3: "High Flood",
            4: "Severe Flood"
        }
        
        # Depth mapping
        self.depth_map = {
            0: ("0-5 cm", 5),
            1: ("5-20 cm", 15),
            2: ("20-50 cm", 35),
            3: ("50-80 cm", 65),
            4: ("80+ cm", 100)
        }
    
    def predict(self, image_path):
        """
        Predict severity from image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Prediction results
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Cannot load image: {e}"}
        
        # Preprocess
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(torch.max(probs).item())
        
        # Get depth info
        depth_band, depth_cm = self.depth_map[pred]
        
        return {
            "image_path": str(image_path),
            "severity_class": pred,
            "severity_name": self.severity_names[pred],
            "confidence": round(confidence, 4),
            "depth_band": depth_band,
            "depth_cm": depth_cm,
            "all_probabilities": {
                self.severity_names[i]: float(probs[0][i].item())
                for i in range(5)
            }
        }


def main():
    """
    Example: Predict single image
    """
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    predictor = SeverityPredictor()
    result = predictor.predict(image_path)
    
    print("\n" + "="*60)
    print("FLOOD SEVERITY PREDICTION")
    print("="*60)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Image: {result['image_path']}")
        print(f"Severity: {result['severity_class']} - {result['severity_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Depth Band: {result['depth_band']}")
        print(f"Estimated Depth: {result['depth_cm']} cm")
        
        print("\nAll Probabilities:")
        for severity_name, prob in result['all_probabilities'].items():
            print(f"  {severity_name}: {prob:.4f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
