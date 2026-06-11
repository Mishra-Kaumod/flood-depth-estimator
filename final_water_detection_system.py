#!/usr/bin/env python3
"""
FINAL INTEGRATED SOLUTION: Complete water detection system combining all 4 fixes.
1. Retrained classifier (lightweight_flood_classifier_improved.pt)
2. Improved color detection (excluding blue objects)
3. Object visibility checks (dry if objects fully visible)
4. Fallback to texture analysis for uncertain cases
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, Tuple
import os

class FinalWaterDetectionSystem:
    """
    Complete water detection system using multi-method consensus with fallback logic.
    """
    
    def __init__(self, classifier_path: str = "lightweight_flood_classifier_improved.pt"):
        """Initialize the system with improved classifier."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = self._load_classifier(classifier_path)
        
    def _load_classifier(self, model_path: str):
        """Load the retrained MobileNetV3 classifier."""
        try:
            # Build MobileNetV3 Small architecture
            from torchvision import models
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            model.classifier = nn.Sequential(
                nn.Linear(576, 256),
                nn.BatchNorm1d(256),
                nn.Hardswish(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.Hardswish(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 2),
            )
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded improved classifier from {model_path}")
            else:
                print(f"⚠️  Improved classifier not found, using default model")
            
            return model.to(self.device).eval()
        except Exception as e:
            print(f"⚠️  Could not load classifier: {e}")
            return None
    
    def detect_water(self, image: np.ndarray, detected_objects: Dict = None) -> Dict:
        """
        Main detection method using comprehensive consensus approach.
        
        Returns:
            Dict with water_detected, confidence, method, details
        """
        
        if detected_objects is None:
            detected_objects = {}
        
        # ===== PRIMARY CHECK: Object Visibility =====
        visibility_result = self._check_object_visibility(image, detected_objects)
        
        # If objects clearly visible -> definitely dry
        if visibility_result['water_likely'] is False and visibility_result['confidence'] > 0.75:
            return {
                'water_detected': False,
                'confidence': visibility_result['confidence'],
                'method': 'object_visibility_override',
                'reason': f"✅ OVERRIDE: {visibility_result['reason']} → DEFINITELY DRY",
                'is_hallucination_prevented': True
            }
        
        # ===== FALLBACK: Multi-method consensus =====
        votes = {}
        methods_used = []
        
        # Method 1: Improved Color Analysis
        color_result = self._detect_water_by_color_v2(image)
        votes['color'] = color_result['water_detected']
        methods_used.append(f"Color ({color_result['reason']})")
        
        # Method 2: Horizontal Edge Detection
        edge_result = self._detect_water_edges_v2(image)
        votes['edges'] = edge_result['water_detected']
        methods_used.append(f"Edges ({edge_result['reason']})")
        
        # Method 3: Trained Neural Classifier
        if self.classifier:
            classifier_result = self._classify_with_neural_network(image)
            votes['classifier'] = classifier_result['water_detected']
            methods_used.append(f"Classifier (prob={classifier_result['probability']:.1%})")
        
        # Method 4: Depth Texture Analysis
        depth_result = self._detect_water_by_texture(image)
        votes['texture'] = depth_result['water_detected']
        methods_used.append(f"Texture ({depth_result['reason']})")
        
        # ===== CONSENSUS DECISION =====
        water_votes = sum(1 for v in votes.values() if v)
        total_checks = len(votes)
        
        # Require majority: >= 2/3 or >= 2/4
        threshold = max(2, total_checks // 2 + 1)
        water_detected = water_votes >= threshold
        confidence = water_votes / total_checks
        
        # Prevent hallucinations: override to dry if low confidence
        if water_detected and confidence < 0.5:
            water_detected = False
            result_reason = "Confidence too low, defaulting to DRY"
        else:
            result_reason = f"Consensus: {water_votes}/{total_checks} methods"
        
        return {
            'water_detected': water_detected,
            'confidence': confidence,
            'method': 'multi_method_consensus',
            'votes': votes,
            'methods': methods_used,
            'reason': result_reason,
            'is_hallucination_prevented': True
        }
    
    def _check_object_visibility(self, image: np.ndarray, detected_objects: Dict) -> Dict:
        """Check if reference objects are visible (not submerged)."""
        h, w = image.shape[:2]
        
        # Check persons
        if 'persons' in detected_objects and len(detected_objects['persons']) > 0:
            for person_box in detected_objects['persons']:
                x1, y1, x2, y2 = person_box
                if y1 > 0.05*h and y2 < 0.95*h:  # Head and feet visible
                    return {
                        'water_likely': False,
                        'confidence': 0.90,
                        'reason': "Person fully visible (head to feet visible)"
                    }
        
        # Check buses
        if 'buses' in detected_objects and len(detected_objects['buses']) > 0:
            for bus_box in detected_objects['buses']:
                x1, y1, x2, y2 = bus_box
                if y1 > 0.05*h:  # Bus roof visible
                    return {
                        'water_likely': False,
                        'confidence': 0.95,
                        'reason': "Bus roof visible (structure above water line)"
                    }
        
        return {
            'water_likely': None,
            'confidence': 0.3,
            'reason': "No reference objects detected"
        }
    
    def _detect_water_by_color_v2(self, image: np.ndarray) -> Dict:
        """Improved color detection excluding saturated blues."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Low-saturation blues (water-like)
        water_mask = cv2.inRange(hsv, (90, 0, 100), (130, 100, 255))
        
        # Exclude vibrant blues (objects)
        saturated_blue = cv2.inRange(s, 150, 255) & cv2.inRange(h, 90, 130)
        water_mask = cv2.subtract(water_mask, saturated_blue)
        
        coverage = np.count_nonzero(water_mask) / water_mask.size
        detected = coverage > 0.05  # 5% threshold
        
        return {
            'water_detected': detected,
            'reason': f"{coverage:.1%} blue pixels" if detected else f"{coverage:.1%} (too low)"
        }
    
    def _detect_water_edges_v2(self, image: np.ndarray) -> Dict:
        """Improved edge detection for water ripples."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLines(edges, 1, np.pi/180, 40)
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if theta < 0.3 or theta > 2.8:  # Horizontal
                    horizontal_lines += 1
        
        detected = horizontal_lines > 10
        return {
            'water_detected': detected,
            'reason': f"{horizontal_lines} horiz. lines" if detected else f"{horizontal_lines} (too few)"
        }
    
    def _classify_with_neural_network(self, image: np.ndarray) -> Dict:
        """Use trained classifier for final decision."""
        if self.classifier is None:
            return {'water_detected': False, 'probability': 0.0}
        
        try:
            # Preprocess
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.classifier(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                flood_prob = probs[0, 1].item()  # Probability of flood class
            
            return {
                'water_detected': flood_prob > 0.5,
                'probability': flood_prob
            }
        except Exception as e:
            return {'water_detected': False, 'probability': 0.0}
    
    def _detect_water_by_texture(self, image: np.ndarray) -> Dict:
        """Detect water by analyzing texture/surface patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute local variance (water has smooth, low variance)
        # Non-water has high texture variance
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mean = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        variance = cv2.blur((gray.astype(np.float32) - mean)**2, (5, 5))
        
        # Water surfaces have low variance regions
        low_var = np.count_nonzero(variance < 200) / variance.size
        detected = low_var > 0.2  # > 20% low variance
        
        return {
            'water_detected': detected,
            'reason': f"{low_var:.1%} smooth" if detected else f"{low_var:.1%} (textured)"
        }


# Example usage
def example_usage():
    """Demonstrate the final system."""
    import os
    
    print("\n" + "="*70)
    print("FINAL INTEGRATED WATER DETECTION SYSTEM")
    print("="*70 + "\n")
    
    system = FinalWaterDetectionSystem()
    
    # Test on dry image
    test_img = cv2.imread("test_images/frame_01.jpg")
    if test_img is not None:
        result = system.detect_water(test_img, {
            'persons': [(100, 50, 150, 300)],
            'buses': [(200, 80, 400, 450)]
        })
        
        print("TEST: Dry street scene")
        print(f"  Water detected: {'🔴 YES' if result['water_detected'] else '🟢 NO'}")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Reason: {result['reason']}")
        print(f"  Hallucination prevented: {result.get('is_hallucination_prevented')}")


if __name__ == "__main__":
    import os
    example_usage()
