#!/usr/bin/env python3
"""
Improved water detection with object visibility analysis to prevent hallucinations.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple

class ImprovedWaterDetector:
    """
    Smart water detection that prevents false positives on dry scenes
    by checking if reference objects are fully visible (not submerged).
    """
    
    def __init__(self):
        self.person_height_cm = 175
        self.bus_height_cm = 300
        self.car_height_cm = 150
        self.motorcycle_height_cm = 100
        self.truck_height_cm = 250
        
    def detect_water_improved(self, image: np.ndarray, 
                            detected_objects: Dict,
                            depth_map: np.ndarray = None) -> Dict:
        """
        Improved water detection with multiple checks to prevent hallucinations.
        
        Args:
            image: Input image (BGR)
            detected_objects: Dict with 'persons', 'cars', 'buses', 'motorcycles', 'trucks'
            depth_map: Optional depth map from Depth Anything V2
        
        Returns:
            Dict with water_detected, confidence, reasons
        """
        
        votes = {}
        reasons = []
        
        # ===== CHECK 1: Object Visibility Analysis =====
        # Most reliable: if reference objects are fully visible (head visible, no water), it's dry
        visibility_check = self._check_object_visibility(image, detected_objects)
        votes['object_visibility'] = visibility_check['water_likely']
        reasons.append(visibility_check['reason'])
        
        if visibility_check['confidence'] > 0.8:
            # High confidence that it's dry based on object visibility
            if not visibility_check['water_likely']:
                return {
                    'water_detected': False,
                    'confidence': visibility_check['confidence'],
                    'method': 'object_visibility',
                    'reasons': reasons,
                    'is_hallucination': False
                }
        
        # ===== CHECK 2: Color Analysis (improved) =====
        color_check = self._detect_water_by_color_improved(image)
        votes['color'] = color_check['water_detected']
        reasons.append(color_check['reason'])
        
        # ===== CHECK 3: Horizontal Edge Lines =====
        edge_check = self._detect_horizontal_edges_improved(image)
        votes['edges'] = edge_check['water_detected']
        reasons.append(edge_check['reason'])
        
        # ===== CHECK 4: Depth Continuity (if available) =====
        if depth_map is not None:
            depth_check = self._detect_depth_discontinuity(depth_map)
            votes['depth'] = depth_check['water_detected']
            reasons.append(depth_check['reason'])
        
        # ===== CONSENSUS VOTING =====
        water_votes = sum(1 for v in votes.values() if v)
        total_checks = len(votes)
        
        # If object visibility indicates dry with high confidence, that's definitive
        if visibility_check['water_likely'] is False and visibility_check['confidence'] > 0.7:
            water_detected = False
            consensus_reason = f"Object visibility OVERRIDE: Objects fully visible → DEFINITELY DRY"
        else:
            # For texture patches or no objects: need at least 2/3 or 2/4 votes
            threshold = 2 if total_checks >= 3 else 1
            water_detected = water_votes >= threshold
            consensus_reason = f"Consensus: {water_votes}/{total_checks} methods detected water"
        
        reasons.append(consensus_reason)
        
        return {
            'water_detected': water_detected,
            'confidence': min(water_votes / total_checks * 0.8 + 0.2, 1.0),  # 0.2 - 1.0
            'votes': votes,
            'reasons': reasons,
            'is_hallucination': False
        }
    
    def _check_object_visibility(self, image: np.ndarray, 
                                detected_objects: Dict) -> Dict:
        """
        Check if reference objects are fully visible and not submerged.
        If person/bus head is visible -> likely dry (not flooded).
        """
        h, w = image.shape[:2]
        
        # Look for people bounding boxes near bottom (not fully submerged)
        if 'persons' in detected_objects and len(detected_objects['persons']) > 0:
            persons = detected_objects['persons']
            for person_box in persons:
                x1, y1, x2, y2 = person_box
                height_pixels = y2 - y1
                
                # If person is fully visible from head to feet in frame, likely dry
                if y1 > 0 and y2 < h * 0.95:  # Head and feet both visible
                    return {
                        'water_likely': False,
                        'confidence': 0.85,
                        'reason': f"Person fully visible in frame (head at y={y1}, feet at y={y2}) - likely DRY"
                    }
                
                # If person is cut off at top (head not visible), might be submerged
                if y1 < h * 0.05 and (y2 - y1) > h * 0.4:
                    return {
                        'water_likely': True,
                        'confidence': 0.6,
                        'reason': f"Person partially visible (head cut off) - possibly FLOODED"
                    }
        
        # Look for buses - check if top of bus is visible
        if 'buses' in detected_objects and len(detected_objects['buses']) > 0:
            buses = detected_objects['buses']
            for bus_box in buses:
                x1, y1, x2, y2 = bus_box
                
                # If bus top is visible in frame, bus is not completely submerged
                if y1 > h * 0.05:  # Bus roof visible
                    return {
                        'water_likely': False,
                        'confidence': 0.9,
                        'reason': f"Bus roof visible (y={y1}) - likely DRY"
                    }
        
        # Check for cars
        if 'cars' in detected_objects and len(detected_objects['cars']) > 0:
            cars = detected_objects['cars']
            for car_box in cars:
                x1, y1, x2, y2 = car_box
                
                if y1 > h * 0.1:
                    return {
                        'water_likely': False,
                        'confidence': 0.8,
                        'reason': f"Car roof visible (y={y1}) - likely DRY"
                    }
        
        return {
            'water_likely': None,
            'confidence': 0.3,
            'reason': "No reference objects detected"
        }
    
    def _detect_water_by_color_improved(self, image: np.ndarray) -> Dict:
        """
        Improved color-based water detection that excludes blue objects.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Water: Blue (H: 90-130) with LOW saturation (washed out)
        # Blue object: Blue (H: 90-130) with HIGH saturation (vibrant)
        water_mask = cv2.inRange(hsv, (90, 0, 100), (130, 100, 255))  # Low saturation blue
        
        # Exclude highly saturated blues (buses, signs)
        saturated_mask = cv2.inRange(s, 150, 255)
        blue_hue = cv2.inRange(h, 90, 130)
        saturated_blue = cv2.bitwise_and(blue_hue, saturated_mask)
        
        # Remove saturated blues from water detection
        water_mask = cv2.subtract(water_mask, saturated_blue)
        
        water_coverage = np.count_nonzero(water_mask) / water_mask.size
        
        # Lowered threshold from 15% to 8% for better detection of water textures
        water_detected = water_coverage > 0.08
        
        return {
            'water_detected': water_detected,
            'coverage': water_coverage,
            'reason': f"Color analysis: {water_coverage:.1%} water pixels (threshold: 8%) - {'DETECTED' if water_detected else 'NOT DETECTED'}"
        }
    
    def _detect_horizontal_edges_improved(self, image: np.ndarray) -> Dict:
        """
        Improved edge detection that distinguishes water ripples from structural lines.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal lines using HoughLines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # Horizontal lines have theta close to 0 or π
                if theta < 0.3 or theta > 2.8:
                    horizontal_lines += 1
        
        # Lowered threshold from 40 to 15 lines for texture detection
        water_detected = horizontal_lines > 15
        
        return {
            'water_detected': water_detected,
            'line_count': horizontal_lines,
            'reason': f"Edge detection: {horizontal_lines} horizontal lines (threshold: 15) - {'DETECTED' if water_detected else 'NOT DETECTED'}"
        }
    
    def _detect_depth_discontinuity(self, depth_map: np.ndarray) -> Dict:
        """
        Detect water by large depth gradient discontinuities.
        Water surface creates sharp depth boundaries.
        """
        # Compute gradient magnitude
        sobelx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Threshold on high gradients
        high_gradient = gradient > np.percentile(gradient, 85)
        coverage = np.count_nonzero(high_gradient) / high_gradient.size
        
        water_detected = coverage > 0.1
        
        return {
            'water_detected': water_detected,
            'gradient_coverage': coverage,
            'reason': f"Depth discontinuity: {coverage:.1%} high gradients (threshold: 10%) - {'DETECTED' if water_detected else 'NOT DETECTED'}"
        }


def test_improved_detector():
    """Test the improved detector on dry vs wet scenarios."""
    print("\n" + "="*70)
    print("TESTING IMPROVED WATER DETECTOR")
    print("="*70 + "\n")
    
    detector = ImprovedWaterDetector()
    
    # Simulate dry scene with visible person and bus
    dry_test = detector.detect_water_improved(
        np.zeros((480, 640, 3), dtype=np.uint8),
        {
            'persons': [(100, 50, 150, 300)],  # Person fully visible
            'buses': [(200, 80, 400, 450)]     # Bus top visible
        }
    )
    
    print("TEST 1: Dry scene (objects fully visible)")
    print(f"  Water Detected: {dry_test['water_detected']}")
    print(f"  Confidence: {dry_test['confidence']:.2f}")
    print(f"  Reasons: {dry_test['reasons']}\n")
    
    # Simulate potentially flooded scene
    flooded_test = detector.detect_water_improved(
        np.zeros((480, 640, 3), dtype=np.uint8),
        {
            'persons': [(100, 10, 150, 250)],  # Person head cut off (likely submerged)
            'buses': []
        }
    )
    
    print("TEST 2: Potentially flooded (person partially visible)")
    print(f"  Water Detected: {flooded_test['water_detected']}")
    print(f"  Confidence: {flooded_test['confidence']:.2f}")
    print(f"  Reasons: {flooded_test['reasons']}\n")


if __name__ == "__main__":
    test_improved_detector()
