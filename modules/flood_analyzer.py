"""Shared flood-analysis pipeline for a still image or a video frame."""

from depth_band_estimator import estimate_depth
from hybrid_depth_estimator import HybridDepthEstimator
from predict_image import SeverityPredictor
from water_detection import WaterDetectionAnalyzer


def flood_level_from_depth(depth_cm, water_detected=True):
    """Return a user-facing flood level from the final estimated depth."""
    if not water_detected:
        return 'No Flood Detected'
    if depth_cm <= 5:
        return 'Very Low Flood'
    if depth_cm <= 20:
        return 'Minor Flood'
    if depth_cm <= 50:
        return 'Moderate Flood'
    if depth_cm <= 80:
        return 'High Flood'
    return 'Severe Flood'


class FloodAnalyzer:
    """Detect water first, then run severity and depth estimation only when needed."""

    def __init__(self, model_path="severity_model.pth", use_hybrid=True):
        self.model_path = model_path
        self.use_hybrid = use_hybrid
        self.water_detector = WaterDetectionAnalyzer()
        self.predictor = None
        self.depth_estimator = None

    def analyze_bgr(self, image, image_path="<in-memory image>"):
        """Analyze one BGR OpenCV image and return a consistent result dictionary."""
        water_result = self.water_detector.detect_water_surface(image)
        result = {
            'image_path': str(image_path),
            'water_detected': water_result['water_detected'],
            'water_confidence': water_result['confidence'],
            'water_percentage': water_result['water_percentage'],
            'water_mask': water_result['water_mask'],
            'method_votes': water_result['method_votes'],
            'severity_class': None,
            'severity_name': 'N/A',
            'severity_confidence': None,
            'depth_band': 'No Water',
            'depth_cm': 0,
            'depth_method': 'No water detected',
            'depth_details': {},
            'final_flood_level': 'No Flood Detected',
            'all_probabilities': {},
        }

        # This is the early exit shared by photos and video frames.  No ML
        # model is loaded or invoked for inputs that do not contain water.
        if not result['water_detected']:
            return result

        try:
            if self.predictor is None:
                self.predictor = SeverityPredictor(model_path=self.model_path)
            prediction = self.predictor.predict_bgr(image, image_path)
            if 'error' in prediction:
                result['error'] = prediction['error']
                return result

            result.update({
                'severity_class': prediction['severity_class'],
                'severity_name': prediction['severity_name'],
                'severity_confidence': prediction['confidence'],
                'all_probabilities': prediction['all_probabilities'],
            })

            if self.use_hybrid:
                try:
                    if self.depth_estimator is None:
                        self.depth_estimator = HybridDepthEstimator()
                    depth = self.depth_estimator.estimate_depth(
                        image=image,
                        water_detected=True,
                        water_percentage=result['water_percentage'],
                        severity_class=result['severity_class'],
                        severity_confidence=result['severity_confidence'],
                        water_mask=result['water_mask'],
                    )
                    result.update({
                        'depth_band': depth['depth_band'],
                        'depth_cm': depth['depth_cm'],
                        'depth_method': depth['method'],
                        'depth_details': depth.get('details', {}),
                        'final_flood_level': flood_level_from_depth(depth['depth_cm']),
                    })
                    return result
                except Exception as exc:
                    # YOLO is optional. Keep the core pipeline usable when a
                    # detector model is unavailable and fall back below.
                    self.use_hybrid = False
                    result['hybrid_warning'] = str(exc)

            depth = estimate_depth(result['severity_class'])
            result.update({
                'depth_band': depth['depth_band'],
                'depth_cm': depth['depth_cm'],
                'depth_method': 'Severity-based',
                'final_flood_level': flood_level_from_depth(depth['depth_cm']),
            })
        except Exception as exc:
            result['error'] = str(exc)

        return result
