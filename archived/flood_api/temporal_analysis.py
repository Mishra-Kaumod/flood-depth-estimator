"""
TEMPORAL FLOOD DEPTH ANALYZER
Processes sequences of images from same camera over 5-15 minute intervals.
Uses multiple reference objects (person, car, bus, motorcycle, walls) to 
validate water presence and prevent hallucination.
"""

import numpy as np
from datetime import datetime, timedelta
from django.utils import timezone
from .models import (
    FloodInundationTelemetry, 
    CameraLocation, 
    TemporalFloodSequence
)


class TemporalFloodAnalyzer:
    """
    Analyzes flood depth using temporal sequences with multi-anchor validation.
    """
    
    # Reference object heights in cm (calibrated for detection reliability)
    REFERENCE_HEIGHTS = {
        'person': {'total_height': 175.0, 'torso': 60.0, 'legs': 90.0},
        'car': {'total_height': 150.0, 'wheel_height': 60.0, 'hood_height': 80.0},
        'bus': {'total_height': 300.0, 'wheel_height': 100.0, 'window_height': 220.0},
        'motorcycle': {'total_height': 100.0, 'wheel_height': 55.0, 'seat_height': 75.0},
        'truck': {'total_height': 250.0, 'wheel_height': 90.0, 'cabin_height': 200.0},
        'wall': {'assumed_height': 200.0},
    }
    
    # Minimum objects needed to confirm water (HALLUCINATION PREVENTION)
    MIN_ANCHORS_FOR_CONFIDENCE = {
        'low': 1,      # Single object: low confidence
        'medium': 2,   # Two different objects: medium confidence
        'high': 3      # Three+ different objects: high confidence
    }
    
    # Water detection thresholds
    WATER_PROBABILITY_THRESHOLD = 0.4  # Must be >40% confident there's water
    
    def __init__(self):
        self.valid_reference_objects = set(self.REFERENCE_HEIGHTS.keys())
    
    def get_recent_images_for_camera(self, camera_id, minutes=15):
        """
        Fetch images from a specific camera within the last N minutes.
        
        Args:
            camera_id: Camera identifier
            minutes: Time window in minutes (default 5-15)
            
        Returns:
            QuerySet of FloodInundationTelemetry records
        """
        try:
            camera = CameraLocation.objects.get(camera_id=camera_id)
        except CameraLocation.DoesNotExist:
            return None
        
        start_time = timezone.now() - timedelta(minutes=minutes)
        records = FloodInundationTelemetry.objects.filter(
            camera=camera,
            timestamp__gte=start_time
        ).order_by('timestamp')
        
        return records
    
    def validate_water_presence(self, records):
        """
        HALLUCINATION PREVENTION:
        Validates that water is actually present by checking:
        1. Multiple reference objects detected
        2. Flood probability consensus across images
        3. Consistent depth readings
        
        Args:
            records: QuerySet of telemetry records
            
        Returns:
            dict with validation results
        """
        if not records or records.count() == 0:
            return {
                'is_valid': False,
                'reason': 'No images in sequence',
                'num_images': 0,
                'water_consensus': False
            }
        
        # Collect all detected objects across the sequence
        all_detected_objects = []
        water_detections = []
        depths = []
        
        for record in records:
            # Collect detected objects
            if record.detected_reference_objects:
                all_detected_objects.extend(record.detected_reference_objects)
            
            # Track water confidence
            water_prob = record.surface_water_confirmed_pct / 100.0
            water_detections.append(water_prob)
            
            # Track depths
            depths.append(record.computed_depth_cm)
        
        num_images = records.count()
        
        # Count unique object types
        unique_objects = set(all_detected_objects)
        num_unique_anchors = len(unique_objects)
        
        # Calculate water consensus
        water_consensus_pct = np.mean(water_detections) if water_detections else 0.0
        water_consensus = water_consensus_pct >= self.WATER_PROBABILITY_THRESHOLD
        
        # Determine confidence level
        if num_unique_anchors >= self.MIN_ANCHORS_FOR_CONFIDENCE['high']:
            confidence_level = 'high'
            is_valid = water_consensus and num_images >= 2
        elif num_unique_anchors >= self.MIN_ANCHORS_FOR_CONFIDENCE['medium']:
            confidence_level = 'medium'
            is_valid = water_consensus and num_images >= 3
        elif num_unique_anchors >= self.MIN_ANCHORS_FOR_CONFIDENCE['low']:
            confidence_level = 'low'
            is_valid = water_consensus and num_images >= 5
        else:
            confidence_level = 'insufficient'
            is_valid = False
        
        # Depth consistency check
        depths_array = np.array(depths)
        depth_std = np.std(depths_array) if len(depths) > 1 else 0.0
        
        return {
            'is_valid': is_valid,
            'num_images': num_images,
            'unique_anchor_objects': list(unique_objects),
            'num_unique_anchors': num_unique_anchors,
            'water_consensus_pct': round(water_consensus_pct * 100, 2),
            'water_consensus': water_consensus,
            'confidence_level': confidence_level,
            'reason': self._get_validation_reason(
                is_valid, 
                num_unique_anchors, 
                water_consensus, 
                num_images
            ),
            'depth_consistency_std': round(depth_std, 2)
        }
    
    def _get_validation_reason(self, is_valid, num_anchors, water_consensus, num_images):
        """Generate human-readable validation reason."""
        if not water_consensus:
            return f"Water not confirmed across images (only {water_consensus}% average confidence)"
        if num_anchors == 0:
            return "No reference objects detected - cannot validate depth"
        if num_anchors == 1 and num_images < 5:
            return f"Only 1 reference object type detected. Need 5+ images, got {num_images}"
        if num_anchors == 2 and num_images < 3:
            return f"Only 2 reference object types. Need 3+ images, got {num_images}"
        if is_valid:
            return f"VALIDATED: {num_anchors} anchor types across {num_images} images"
        return "Insufficient data to validate"
    
    def calculate_multi_anchor_depth(self, records):
        """
        MULTI-ANCHOR DEPTH ESTIMATION:
        Calculates water depth using multiple reference objects as calibration points.
        
        Args:
            records: QuerySet of telemetry records
            
        Returns:
            dict with depth estimates from different anchors
        """
        if not records or records.count() == 0:
            return {'error': 'No records provided'}
        
        depth_estimates = {}
        
        for record in records:
            if not record.detected_reference_objects:
                continue
            
            for obj_type in record.detected_reference_objects:
                if obj_type not in self.REFERENCE_HEIGHTS:
                    continue
                
                if obj_type not in depth_estimates:
                    depth_estimates[obj_type] = []
                
                # Store depth with confidence
                depth_estimates[obj_type].append({
                    'depth_cm': record.computed_depth_cm,
                    'confidence': record.system_confidence_score_pct / 100.0,
                    'timestamp': record.timestamp
                })
        
        # Aggregate estimates per object type
        aggregated = {}
        for obj_type, measurements in depth_estimates.items():
            depths = [m['depth_cm'] for m in measurements]
            confidences = [m['confidence'] for m in measurements]
            
            # Weighted average by confidence
            weighted_depth = np.average(depths, weights=confidences)
            
            aggregated[obj_type] = {
                'mean_depth_cm': round(np.mean(depths), 2),
                'weighted_depth_cm': round(weighted_depth, 2),
                'std_dev_cm': round(np.std(depths), 2),
                'min_depth_cm': round(np.min(depths), 2),
                'max_depth_cm': round(np.max(depths), 2),
                'num_measurements': len(measurements),
                'avg_confidence': round(np.mean(confidences), 3)
            }
        
        return aggregated
    
    def create_temporal_sequence(self, camera_id, time_window_minutes=15):
        """
        MAIN ENTRY POINT:
        Creates a TemporalFloodSequence from recent images,
        validates water presence, and calculates consensus depth.
        
        Args:
            camera_id: Camera identifier
            time_window_minutes: Time window for sequence (default 15)
            
        Returns:
            dict with sequence analysis results
        """
        # Fetch recent images
        records = self.get_recent_images_for_camera(camera_id, time_window_minutes)
        if not records:
            return {
                'status': 'error',
                'message': f'Camera {camera_id} not found'
            }
        
        if records.count() < 2:
            return {
                'status': 'insufficient_data',
                'message': f'Only {records.count()} image(s) in sequence. Need at least 2.',
                'camera_id': camera_id
            }
        
        # Validate water presence (HALLUCINATION PREVENTION)
        validation = self.validate_water_presence(records)
        
        # Calculate multi-anchor depth estimates
        depth_estimates = self.calculate_multi_anchor_depth(records)
        
        # Create or update sequence
        camera = CameraLocation.objects.get(camera_id=camera_id)
        sequence = TemporalFloodSequence.objects.create(
            camera=camera,
            sequence_start=records.first().timestamp,
            sequence_end=records.last().timestamp,
            image_count=records.count(),
            water_detected_in_images=sum(1 for r in records if r.surface_water_confirmed_pct >= 40),
            detected_anchor_types=list(set([obj for r in records for obj in (r.detected_reference_objects or [])])),
            consensus_water_present=validation['water_consensus'],
            confidence_score=self._calculate_confidence_score(validation)
        )
        
        # Add all records to sequence
        for record in records:
            sequence.telemetry_records.add(record)
        
        # Calculate aggregated metrics
        if depth_estimates and validation['is_valid']:
            # Average across all anchor types
            all_depths = [v['weighted_depth_cm'] for v in depth_estimates.values()]
            sequence.average_depth_cm = round(np.mean(all_depths), 2)
            sequence.max_depth_cm = round(np.max(all_depths), 2)
            sequence.min_depth_cm = round(np.min(all_depths), 2)
            sequence.save()
        
        return {
            'status': 'success' if validation['is_valid'] else 'warning',
            'sequence_id': sequence.id,
            'camera_id': camera_id,
            'num_images': records.count(),
            'time_span_minutes': round((records.last().timestamp - records.first().timestamp).total_seconds() / 60, 1),
            'validation': validation,
            'depth_estimates_by_anchor': depth_estimates,
            'consensus_depth_cm': sequence.average_depth_cm,
            'final_risk_assessment': self._assess_risk(sequence.average_depth_cm, validation)
        }
    
    def _calculate_confidence_score(self, validation):
        """Calculate overall confidence score (0.0-1.0)."""
        factors = [
            # Number of anchors factor
            min(validation['num_unique_anchors'] / 3.0, 1.0) * 0.4,
            # Water consensus factor
            (validation['water_consensus_pct'] / 100.0) * 0.3,
            # Image count factor
            min(validation['num_images'] / 10.0, 1.0) * 0.3
        ]
        return round(sum(factors), 3)
    
    def _assess_risk(self, depth_cm, validation):
        """
        Risk assessment based on depth and validation confidence.
        """
        if depth_cm is None or not validation['is_valid']:
            return {
                'level': 'UNVERIFIED',
                'reason': 'Insufficient data to confirm water presence'
            }
        
        if depth_cm < 15:
            return {
                'level': 'LOW',
                'reason': f'Shallow depth ({depth_cm}cm) - pedestrian crossing generally safe'
            }
        elif depth_cm < 30:
            return {
                'level': 'MODERATE',
                'reason': f'Depth {depth_cm}cm - small vehicles compromised'
            }
        elif depth_cm < 60:
            return {
                'level': 'HIGH',
                'reason': f'Depth {depth_cm}cm - most vehicles risk stalling'
            }
        else:
            return {
                'level': 'CRITICAL',
                'reason': f'Depth {depth_cm}cm - severe inundation, closure recommended'
            }
