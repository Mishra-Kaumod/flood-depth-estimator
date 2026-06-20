"""
YOLO OBJECT DETECTION MODULE

Detects vehicles and people in images/frames.
Useful for:
- Object inventory (what's in the scene)
- Anchor-based depth estimation (using known object dimensions)
- Scene understanding for flood analysis
"""

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """
    YOLO-based object detection for vehicles and people.
    """
    
    # Known physical dimensions (in cm)
    OBJECT_SPECS = {
        "car": {"wheel_diameter": 65, "height": 145, "width": 175},
        "truck": {"wheel_diameter": 90, "height": 200, "width": 250},
        "bus": {"wheel_diameter": 100, "height": 280, "width": 260},
        "motorcycle": {"wheel_diameter": 56, "height": 90, "width": 80},
        "bicycle": {"wheel_diameter": 56, "height": 100, "width": 70},
        "person": {"height": 170, "width": 45},
    }
    
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initialize YOLO object detector.
        
        Args:
            model_name: YOLOv8 model variant
                - yolov8n.pt: nano (fastest, lower accuracy)
                - yolov8s.pt: small
                - yolov8m.pt: medium
                - yolov8l.pt: large
                - yolov8x.pt: xlarge (slowest, best accuracy)
        """
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model {model_name}: {e}")
    
    def detect_objects(self, image):
        """
        Detect objects in image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            list: Detected objects with properties
        """
        try:
            results = self.model(image, verbose=False)[0]
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            return []
        
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            # Get coordinates
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            confidence = float(box.conf[0])
            
            # Calculate dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Get physical specs if available
            specs = self.OBJECT_SPECS.get(class_name, {})
            
            detection = {
                'class': class_name,
                'confidence': confidence,
                'bbox': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': box_width,
                    'height': box_height
                },
                'specs': specs,
                'area_pixels': box_width * box_height
            }
            
            detections.append(detection)
        
        return detections
    
    def get_largest_object(self, detections):
        """
        Get the largest detected object (by pixel area).
        
        Args:
            detections: List of detection results
            
        Returns:
            dict: Largest detection or None
        """
        if not detections:
            return None
        
        return max(detections, key=lambda x: x['area_pixels'])
    
    def estimate_depth_from_object(self, detection, image_height):
        """
        Estimate water depth using detected object as reference.
        
        Formula: 
        - Get object height in image (pixels)
        - Get object actual height (cm) from specs
        - Calculate pixels-per-cm ratio
        - Use bottom of bounding box as water level
        - Measure submerged height and convert to cm
        
        Args:
            detection: Single detection result
            image_height: Total image height in pixels
            
        Returns:
            dict: Depth estimation info
        """
        specs = detection.get('specs', {})
        bbox = detection['bbox']
        
        # Check if we have height info for this object
        if 'height' not in specs:
            return {
                'depth_cm': None,
                'method': 'No specs',
                'object': detection['class']
            }
        
        actual_height_cm = specs['height']
        box_height_pixels = bbox['height']
        
        if box_height_pixels == 0:
            return {
                'depth_cm': None,
                'method': 'Invalid box',
                'object': detection['class']
            }
        
        # Pixels per cm for this object
        pixels_per_cm = box_height_pixels / actual_height_cm
        
        # Estimate water level (assume bottom of object touches ground)
        # Water depth = how much of the object is submerged
        object_bottom = bbox['y2']
        
        # Submerged portion (rough estimate from object bottom)
        # Typical: if object_bottom is in lower 30% of image, assume some submersion
        image_bottom = image_height
        submerged_pixels = image_bottom - object_bottom
        
        # Convert to cm
        if submerged_pixels > 0:
            depth_cm = int(submerged_pixels / pixels_per_cm)
        else:
            depth_cm = 0
        
        return {
            'depth_cm': depth_cm,
            'method': f'{detection["class"]}_reference',
            'object': detection['class'],
            'confidence': detection['confidence'],
            'pixels_per_cm': pixels_per_cm,
            'submerged_pixels': submerged_pixels
        }
    
    def draw_detections(self, image, detections, show_specs=False):
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: BGR image to annotate
            detections: List of detection results
            show_specs: Whether to show physical specs
            
        Returns:
            Annotated image
        """
        img = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(
                img,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                (0, 255, 0),  # Green
                2
            )
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            if show_specs and det['specs']:
                specs_text = det['specs'].get('height', '')
                if specs_text:
                    label += f" ({specs_text}cm)"
            
            cv2.putText(
                img, label,
                (bbox['x1'], bbox['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )
            
            # Draw center point
            cv2.circle(img, (bbox['center_x'], bbox['center_y']), 3, (0, 0, 255), -1)
        
        return img
    
    def create_object_inventory(self, detections):
        """
        Create summary of detected objects.
        
        Args:
            detections: List of detection results
            
        Returns:
            dict: Summary stats
        """
        if not detections:
            return {
                'total_objects': 0,
                'object_types': {},
                'avg_confidence': 0
            }
        
        object_types = {}
        total_confidence = 0
        
        for det in detections:
            class_name = det['class']
            confidence = det['confidence']
            
            if class_name not in object_types:
                object_types[class_name] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'confidences': []
                }
            
            object_types[class_name]['count'] += 1
            object_types[class_name]['confidences'].append(confidence)
            total_confidence += confidence
        
        # Calculate averages
        for obj_type in object_types:
            confidences = object_types[obj_type]['confidences']
            object_types[obj_type]['avg_confidence'] = sum(confidences) / len(confidences)
            del object_types[obj_type]['confidences']  # Remove raw list
        
        avg_confidence = total_confidence / len(detections) if detections else 0
        
        return {
            'total_objects': len(detections),
            'object_types': object_types,
            'avg_confidence': round(avg_confidence, 4)
        }


def main():
    """
    Example usage of object detector.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python object_detection.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        sys.exit(1)
    
    # Detect objects
    detector = ObjectDetector()
    detections = detector.detect_objects(image)
    
    # Print results
    print("\n" + "="*60)
    print(f"Detected {len(detections)} objects")
    print("="*60)
    
    for i, det in enumerate(detections, 1):
        print(f"\n{i}. {det['class'].upper()}")
        print(f"   Confidence: {det['confidence']:.2%}")
        print(f"   Bounding Box: x1={det['bbox']['x1']}, y1={det['bbox']['y1']}, "
              f"x2={det['bbox']['x2']}, y2={det['bbox']['y2']}")
        print(f"   Size: {det['bbox']['width']}x{det['bbox']['height']} pixels")
        
        if det['specs']:
            print(f"   Specs: {det['specs']}")
    
    # Get inventory
    inventory = detector.create_object_inventory(detections)
    
    print("\n" + "="*60)
    print("INVENTORY SUMMARY")
    print("="*60)
    print(f"Total Objects: {inventory['total_objects']}")
    print(f"Average Confidence: {inventory['avg_confidence']:.2%}")
    
    if inventory['object_types']:
        print("\nObject Types:")
        for obj_type, info in inventory['object_types'].items():
            print(f"  {obj_type}: {info['count']} (avg confidence: {info['avg_confidence']:.2%})")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
