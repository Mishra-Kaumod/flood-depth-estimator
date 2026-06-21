"""
HYBRID DEPTH ESTIMATOR

Combines multiple approaches for robust depth estimation:
1. Water detection confidence
2. YOLO object detection (for anchor-based estimation)
3. Severity classification
4. Ensemble voting for final depth
"""

from object_detection import ObjectDetector
from depth_band_estimator import DEPTH_BANDS, estimate_depth


class HybridDepthEstimator:
    """
    Multi-method depth estimation combining water detection, 
    object detection, and severity classification.
    """
    
    def __init__(self):
        """Initialize object detector."""
        self.object_detector = ObjectDetector()
    
    def estimate_depth(self, image, water_detected, water_percentage,
                      severity_class, severity_confidence, water_mask=None):
        """
        Estimate water depth using multiple methods.
        
        Args:
            image: BGR image
            water_detected: Boolean - is water present
            water_percentage: Float 0-1 - % of image that is water
            severity_class: Int 0-4 - severity classification
            severity_confidence: Float 0-1 - confidence in classification
            water_mask: Optional binary water mask aligned to ``image``
            
        Returns:
            dict: Depth estimation results
        """
        
        if not water_detected:
            return {
                'water_detected': False,
                'depth_cm': 0,
                'depth_band': 'No Water',
                'method': 'No water detected',
                'confidence': 0.0,
                'details': {}
            }
        
        # Method 1: Severity-based depth
        severity_depth = estimate_depth(severity_class)
        method1_depth_cm = severity_depth['depth_cm']
        method1_band = severity_depth['depth_band']
        # method1_confidence = severity_confidence if severity_confidence else 0
        # Severity model is useful but tends to over/underestimate real depth
        method1_confidence = (
            severity_confidence * 0.5
            if severity_confidence
            else 0
        )
        
        # Method 2: Object-based depth (if objects detected)
        method2_depth_cm = None
        method2_confidence = 0
        object_info = None
        
        try:
            detections = self.object_detector.detect_objects(image)
            
            if detections:
                # Prefer a person over a large vehicle: a visible person's
                # submersion is usually a more informative depth reference.
                largest = self.object_detector.get_best_depth_reference(detections)
                
                if largest:
                    depth_result = self.object_detector.estimate_depth_from_object(
                        largest, image.shape[0], water_mask=water_mask
                    )
                    
                    if depth_result['depth_cm'] is not None:
                        method2_depth_cm = depth_result['depth_cm']
                        method2_confidence = depth_result['confidence']
                        object_info = depth_result
                        # Boost trust in object-based estimates
                        if method2_depth_cm is not None:

                            if method2_depth_cm > 70:
                                method2_confidence *= 2.5

                            elif method2_depth_cm > 40:
                                method2_confidence *= 1.5
                        if method2_confidence > 0.7:
                            method2_confidence *= 2.0
                        
        
        except Exception as e:
            pass  # YOLO optional, continue without it
        
        # Method 3: Water percentage heuristic
        # Higher water percentage suggests deeper flooding
        # if water_percentage > 0.7:
        #     method3_depth_cm = 80
        # elif water_percentage > 0.4:
        #     method3_depth_cm = 50
        if water_percentage > 0.7:
            method3_depth_cm = 25
        elif water_percentage > 0.4:
            method3_depth_cm = 15
        elif water_percentage > 0.2:
            method3_depth_cm = 8
        else:
            method3_depth_cm = 3
        # elif water_percentage > 0.2:
        #     method3_depth_cm = 25
        # else:
        #     method3_depth_cm = 10

        # method3_confidence = min(
        #     0.8,
        #     max(0.2, water_percentage)
        # )
        method3_confidence = min(
            0.25,
            max(0.05, water_percentage*0.25)
        )
        
        
        # Consistency check between severity estimate and object estimate
        if (
            method2_depth_cm is not None
            and method2_depth_cm > 70
            and method1_depth_cm < 20
        ):
            # Object estimate likely more trustworthy
            method1_confidence *= 0.2
        
        # Explicit weighting
        method1_confidence *= 0.5      # severity
        method2_confidence *= 2.0      # object
        method3_confidence *= 0.5      # water coverage
        method1_confidence = min(method1_confidence, 1.0)
        method2_confidence = min(method2_confidence, 1.0)
        method3_confidence = min(method3_confidence, 1.0)
        # Ensemble: Weighted voting
        total_weight = method1_confidence + method2_confidence + method3_confidence
        
        if total_weight > 0:
            final_depth_cm = (
                (method1_depth_cm * method1_confidence) +
                (method2_depth_cm * method2_confidence if method2_depth_cm is not None else 0) +
                (method3_depth_cm * method3_confidence)
            ) / total_weight
        else:
            final_depth_cm = method1_depth_cm  # Fallback to severity
        
        # # final_depth_cm = int(final_depth_cm)
        # if (
        #     method1_depth_cm <= 20
        #     and water_percentage < 0.85
        # ):
        #     final_depth_cm = min(final_depth_cm, 20)
        
        # Get depth band
        final_band = "Unknown"
        for severity, (band, depth) in DEPTH_BANDS.items():
            if final_depth_cm <= depth:
                final_band = band
                break
        
        return {
            'water_detected': True,
            'depth_cm': final_depth_cm,
            'depth_band': final_band,
            'water_percentage': round(water_percentage * 100, 2),
            'method': 'Ensemble (Severity + Object + Water %)',
            # 'confidence': round(total_weight / 3, 4),  # Normalized to 0-1
             'confidence': min(
                 1.0,
                 round(total_weight / 2.0, 4)
                 ),
            'details': {
                'method_1_severity': {
                    'depth_cm': method1_depth_cm,
                    'band': method1_band,
                    'confidence': round(method1_confidence, 4)
                },
                'method_2_object': {
                    'depth_cm': method2_depth_cm,
                    'confidence': round(method2_confidence, 4),
                    'object_info': object_info
                },
                'method_3_water_coverage': {
                    'depth_cm': method3_depth_cm,
                    'confidence': round(method3_confidence, 4)
                }
            }
        }
    
    def get_object_inventory(self, image):
        """
        Get inventory of objects in frame.
        
        Args:
            image: BGR image
            
        Returns:
            dict: Object inventory
        """
        try:
            detections = self.object_detector.detect_objects(image)
            inventory = self.object_detector.create_object_inventory(detections)
            return inventory
        except Exception as e:
            return {
                'error': str(e),
                'total_objects': 0
            }


if __name__ == "__main__":
    # Example usage
    import cv2
    
    estimator = HybridDepthEstimator()
    
    # Example with dummy values
    image = cv2.imread("test_image.jpg")
    
    result = estimator.estimate_depth(
        image=image,
        water_detected=True,
        water_percentage=0.45,
        severity_class=2,
        severity_confidence=0.85
    )
    
    print("\nDepth Estimation Result:")
    print(f"Depth: {result['depth_cm']} cm ({result['depth_band']})")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Method: {result['method']}")
