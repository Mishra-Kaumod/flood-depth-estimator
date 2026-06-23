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
import torch
from ultralytics import YOLO

try:
    from .reference_dimensions import get_object_specs
except ImportError:
    from reference_dimensions import get_object_specs


class ObjectDetector:
    """
    YOLO-based object detection for vehicles and people.
    """

    # Fallback physical dimensions (in cm) for objects not covered by the
    # reference dimensions module.
    FALLBACK_SPECS = {
        "motorcycle": {"wheel_diameter": 56, "height": 90, "width": 80},
        "bicycle": {"wheel_diameter": 56, "height": 100, "width": 70},
    }
  
    def __init__(self, model_name="yolov8n.pt"):
        # yolov8s.pt
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
            self.model = self._load_yolo_model(model_name)
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model {model_name}: {e}")

    def _load_yolo_model(self, model_name):
        """Load YOLO model while forcing torch.load to use weights_only=False."""
        orig_torch_load = torch.load

        def patched_torch_load(f, *args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return orig_torch_load(f, *args, **kwargs)

        torch.load = patched_torch_load
        try:
            return YOLO(model_name)
        finally:
            torch.load = orig_torch_load
    
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
            
            # Get physical specs if available from reference dimensions
            specs = get_object_specs(class_name) or self.FALLBACK_SPECS.get(class_name, {})
            
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

    def get_best_depth_reference(self, detections):
        """Choose a useful calibrated reference, preferring a detected person.

        A person is usually the best flood gauge because its visible height is
        directly reduced by submersion. Vehicles are used only when no person
        with known dimensions is available.
        """
        supported = [d for d in detections if d.get('specs', {}).get('height')]
        if not supported:
            return None

        people = [d for d in supported if d['class'] == 'person']
        candidates = people or supported
        return max(candidates, key=lambda d: d['confidence'] * d['area_pixels'])

    def estimate_depth_from_object(self, detection, image_height, water_mask=None):
        """Estimate depth from an object's visible height and a waterline.

        The previous implementation treated the bottom edge of the *image* as
        ground, which makes depth depend on camera framing.  This version uses
        a waterline inside the detected object region (when a water mask is
        available) and estimates how much of a known-height object is hidden.
        It is an image-based estimate, not a survey-grade measurement.
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
        box_width_pixels = bbox['width']

        if box_height_pixels <= 0 or box_width_pixels <= 0:
            return {
                'depth_cm': None,
                'method': 'Invalid box',
                'object': detection['class']
            }
        
        # Width is normally still visible above the waterline, so it gives a
        # more stable scale than the truncated visible height.
        actual_width_cm = specs.get('width')
        if actual_width_cm:
            pixels_per_cm = box_width_pixels / actual_width_cm
            scale_method = 'object_width'
        else:
            pixels_per_cm = box_height_pixels / actual_height_cm
            scale_method = 'object_height_fallback'

        waterline_y, waterline_source, waterline_confidence = self._find_waterline(
            bbox, water_mask, image_height
        )
        visible_height_pixels = max(0, waterline_y - bbox['y1'])
        visible_height_cm = visible_height_pixels / pixels_per_cm
        depth_cm = int(np.clip(actual_height_cm - visible_height_cm, 0, actual_height_cm * 0.9))
        
        return {
            'depth_cm': depth_cm,
            'method': f'{detection["class"]}_waterline_reference',
            'object': detection['class'],
            'confidence': round(detection['confidence'] * waterline_confidence, 4),
            'pixels_per_cm': pixels_per_cm,
            'waterline_y': waterline_y,
            'waterline_source': waterline_source,
            'waterline_confidence': waterline_confidence,
            'visible_height_cm': round(visible_height_cm, 2),
            'scale_method': scale_method,
            'specs': specs,
        }

    def _find_waterline(self, bbox, water_mask, image_height):
        """Find the top of a stable water region inside a detected object box."""
        fallback_y = min(max(bbox['y2'], 0), image_height - 1)
        if water_mask is None or water_mask.ndim != 2:
            return fallback_y, 'visible_object_bottom', 0.45

        mask_height, mask_width = water_mask.shape[:2]
        x1 = max(0, min(bbox['x1'], mask_width - 1))
        x2 = max(x1 + 1, min(bbox['x2'], mask_width))
        y1 = max(0, min(bbox['y1'], mask_height - 1))
        y2 = max(y1 + 1, min(bbox['y2'], mask_height))
        region = water_mask[y1:y2, x1:x2] > 0
        if region.size == 0:
            return fallback_y, 'visible_object_bottom', 0.45

        row_coverage = region.mean(axis=1)
        top_quarter = row_coverage[:max(1, len(row_coverage) // 4)].mean()
        bottom_quarter = row_coverage[-max(1, len(row_coverage) // 4):].mean()
        min_run = max(3, len(row_coverage) // 8)

        # A valid waterline needs substantially more water-mask support below
        # it than above it; this prevents a noisy all-object mask from being
        # mistaken for a water boundary.
        if bottom_quarter >= top_quarter + 0.15:
            for row in range(max(1, len(row_coverage) // 5), len(row_coverage) - min_run):
                if np.all(row_coverage[row:row + min_run] >= 0.5):
                    confidence = min(0.95, 0.55 + (bottom_quarter - top_quarter) * 0.5)
                    return y1 + row, 'water_mask_transition', round(float(confidence), 4)

        return fallback_y, 'visible_object_bottom', 0.45
    
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
