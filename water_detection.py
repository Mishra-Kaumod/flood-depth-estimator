"""
ADVANCED WATER DETECTION & ANALYSIS ENGINE
Multi-layered approach to detect water surfaces and prevent false positives.

This module analyzes:
1. Visual water signatures (reflections, ripples, color)
2. Surface edge detection 
3. Depth discontinuities
4. Contrast patterns typical of water surfaces
5. Optical flow indicators (water motion)
"""

import cv2
import numpy as np
from scipy import ndimage


class WaterDetectionAnalyzer:
    """
    Advanced water surface detection using multiple computer vision techniques.
    """
    
    def __init__(self):
        self.min_water_area_pct = 0.05  # At least 5% of image should be water
        self.contrast_threshold = 0.15
        self.edge_threshold = 0.1
        
    def detect_water_surface(self, image, depth_map=None):
        """
        Comprehensive water detection using multiple methods.
        
        Args:
            image: BGR image
            depth_map: Optional normalized depth map (0-1)
            
        Returns:
            dict with detection results
        """
        h, w = image.shape[:2]
        
        # Run all detection methods
        results = {
            'rgb_color_analysis': self._detect_water_by_color(image),
            'edge_detection': self._detect_water_edges(image),
            'contrast_analysis': self._detect_water_contrast(image),
            'horizontal_line_detection': self._detect_water_surface_line(image),
            'depth_discontinuity': self._detect_depth_discontinuity(depth_map) if depth_map is not None else None,
            'optical_flow_ripples': self._detect_ripple_patterns(image),
        }
        
        # Aggregate results
        consensus = self._aggregate_detections(results, h, w)
        
        return {
            'water_detected': consensus['water_detected'],
            'confidence': consensus['confidence'],
            'water_percentage': consensus['water_percentage'],
            'method_votes': consensus['method_votes'],
            'details': results,
            'water_mask': consensus['water_mask']
        }
    
    # ========================================================================
    # METHOD 1: Color-based Water Detection
    # ========================================================================
    def _detect_water_by_color(self, image):
        """
        Water surfaces have characteristic color properties:
        - Reflective: low brightness variance horizontally
        - Blue-ish in HSV (depending on sky)
        - Darker than sky due to reflection angles
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Detect cyan/blue-ish water
        # H: 90-130 (blue-green range)
        lower_blue = np.array([90, 20, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Detect dark reflective surfaces (low saturation + medium-low brightness)
        # These are characteristic of water reflections
        dark_reflective = cv2.inRange(s, 0, 100) & cv2.inRange(v, 40, 200)
        
        # Combine masks
        water_color_mask = cv2.bitwise_or(blue_mask, dark_reflective)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_color_mask = cv2.morphologyEx(water_color_mask, cv2.MORPH_CLOSE, kernel)
        
        water_pct = np.count_nonzero(water_color_mask) / (image.shape[0] * image.shape[1])
        
        return {
            'water_detected': water_pct > 0.05,
            'percentage': water_pct,
            'mask': water_color_mask,
            'method': 'Color-based (HSV)'
        }
    
    # ========================================================================
    # METHOD 2: Edge Detection for Water Surface Lines
    # ========================================================================
    def _detect_water_edges(self, image):
        """
        Water surfaces typically have sharp edges between water and objects.
        Use Canny edge detection + Hough line detection.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter to preserve edges while removing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Canny edge detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Detect horizontal lines (water surfaces are typically horizontal)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly horizontal (small vertical difference)
                if abs(y2 - y1) < 20:  # Within 20 pixels vertically
                    horizontal_lines += 1
        
        edge_strength = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
        
        return {
            'water_detected': horizontal_lines >= 3,  # At least 3 horizontal edge lines
            'horizontal_lines': int(horizontal_lines),
            'edge_strength': float(edge_strength),
            'mask': edges,
            'method': 'Horizontal Edge Lines'
        }
    
    # ========================================================================
    # METHOD 3: Contrast Analysis
    # ========================================================================
    def _detect_water_contrast(self, image):
        """
        Water surfaces have specific contrast patterns:
        - Low contrast horizontally (smooth water surface)
        - High contrast at water-object boundaries
        - Distinct bottom-to-top gradient (water darker)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Divide image into horizontal bands and analyze contrast
        band_height = h // 5
        contrast_values = []
        
        for i in range(5):
            band = gray[i*band_height:(i+1)*band_height, :]
            # Laplacian of Gaussian for contrast detection
            contrast = cv2.Laplacian(band, cv2.CV_64F)
            contrast_values.append(np.std(contrast))
        
        # Water shows lower internal contrast, higher edge contrast
        # Expect: [high, lower, lower, lower, variable]
        pattern_match = (
            contrast_values[0] > contrast_values[1] and
            contrast_values[1] < contrast_values[2]
        )
        
        avg_contrast = np.mean(contrast_values)
        
        # Create contrast mask
        contrast_map = cv2.Laplacian(gray, cv2.CV_64F)
        contrast_map = cv2.normalize(contrast_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, contrast_mask = cv2.threshold(contrast_map, 30, 255, cv2.THRESH_BINARY_INV)
        
        return {
            'water_detected': pattern_match,
            'avg_contrast': float(avg_contrast),
            'pattern_match': pattern_match,
            'mask': contrast_mask,
            'method': 'Contrast Pattern Analysis'
        }
    
    # ========================================================================
    # METHOD 4: Horizontal Surface Line Detection
    # ========================================================================
    def _detect_water_surface_line(self, image):
        """
        Water surfaces are typically at a clear horizontal line.
        Use structure tensor to find continuous horizontal features.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Use Sobel to find edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        # Normalize
        sobelx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sobely = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Strong horizontal edges have high sobely, low sobelx
        horizontal_strength = sobely.astype(np.float32)
        
        # Find prominent horizontal lines (average across columns)
        col_prominence = np.mean(horizontal_strength, axis=1)
        
        # Water surface usually in lower half
        lower_half_prominence = col_prominence[h//2:]
        prominent_rows = np.where(lower_half_prominence > np.percentile(lower_half_prominence, 75))[0]
        
        # Check if there's a continuous prominent band
        continuous_band = len(prominent_rows) > 20  # At least 20 pixels of prominence
        max_prominence = float(np.max(col_prominence))
        
        # Create mask
        threshold = np.percentile(horizontal_strength, 80)
        surface_mask = (horizontal_strength > threshold).astype(np.uint8) * 255
        
        return {
            'water_detected': continuous_band and max_prominence > 50,
            'max_prominence': max_prominence,
            'continuous_band': continuous_band,
            'mask': surface_mask,
            'method': 'Horizontal Surface Line'
        }
    
    # ========================================================================
    # METHOD 5: Depth Discontinuity Detection
    # ========================================================================
    def _detect_depth_discontinuity(self, depth_map):
        """
        Water surfaces show characteristic depth patterns:
        - Sharp discontinuities at water-air boundary
        - Smooth variations within water (due to depth)
        """
        if depth_map is None:
            return None
        
        # Compute gradients
        grad_y = np.gradient(depth_map, axis=0)
        grad_x = np.gradient(depth_map, axis=1)
        
        # Magnitude of depth gradient
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Water boundaries show sharp depth changes
        boundary_mask = (grad_mag > np.percentile(grad_mag, 85)).astype(np.uint8) * 255
        
        # Water surface typically shows high horizontal gradient (y-direction)
        horizontal_discontinuity = np.abs(grad_y) > np.percentile(np.abs(grad_y), 80)
        
        h, w = depth_map.shape
        discontinuity_pct = np.count_nonzero(horizontal_discontinuity) / (h * w)
        
        return {
            'water_detected': discontinuity_pct > 0.10,
            'discontinuity_percentage': float(discontinuity_pct),
            'mask': boundary_mask,
            'method': 'Depth Discontinuity'
        }
    
    # ========================================================================
    # METHOD 6: Ripple/Motion Pattern Detection
    # ========================================================================
    def _detect_ripple_patterns(self, image):
        """
        Water surfaces often show ripple or motion artifacts.
        Detect using local variance analysis.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use LBP-style local variance
        h, w = gray.shape
        local_var = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                patch = gray[i-1:i+2, j-1:j+2].astype(np.float32)
                local_var[i, j] = np.var(patch)
        
        # Ripples show moderate variance (not too smooth, not too noisy)
        ripple_mask = (
            (local_var > np.percentile(local_var, 30)) &
            (local_var < np.percentile(local_var, 90))
        ).astype(np.uint8) * 255
        
        ripple_pct = np.count_nonzero(ripple_mask) / (h * w)
        
        return {
            'water_detected': ripple_pct > 0.15,
            'ripple_percentage': float(ripple_pct),
            'mask': ripple_mask,
            'method': 'Ripple Pattern Detection'
        }
    
    # ========================================================================
    # Aggregate Results
    # ========================================================================
    def _aggregate_detections(self, results, h, w):
        """
        Combine results from all methods using voting/consensus.
        """
        votes = 0
        total_methods = 0
        method_votes = {}
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for method_name, result in results.items():
            if result is None:
                continue
            
            total_methods += 1
            detected = result.get('water_detected', False)
            if detected:
                votes += 1
            
            method_votes[method_name] = {
                'detected': detected,
                'confidence': result.get('percentage', result.get('avg_contrast', result.get('discontinuity_percentage', 0)))
            }
            
            # Add mask to combined mask
            if 'mask' in result and result['mask'] is not None:
                combined_mask = cv2.bitwise_or(combined_mask, result['mask'].astype(np.uint8))
        
        # Consensus: at least 3 out of 6 methods vote for water
        water_detected = votes >= 3
        confidence = votes / max(total_methods, 1)  # Normalize to 0-1
        
        water_pct = np.count_nonzero(combined_mask) / (h * w) if h * w > 0 else 0
        
        return {
            'water_detected': water_detected,
            'confidence': float(confidence),
            'water_percentage': float(water_pct),
            'method_votes': method_votes,
            'votes_for_water': votes,
            'total_methods': total_methods,
            'water_mask': combined_mask
        }
    
    def generate_report(self, detection_result):
        """
        Generate human-readable report of water detection analysis.
        """
        result = detection_result
        votes_for = result.get('votes_for_water', 0)
        total = result.get('total_methods', 6)
        
        report = f"""
WATER DETECTION ANALYSIS REPORT
{'='*60}
Overall Result: {'✅ WATER DETECTED' if result['water_detected'] else '❌ NO WATER'}
Confidence: {result['confidence']:.1%}
Water Coverage: {result['water_percentage']:.1%}
Method Consensus: {votes_for}/{total} methods agree

Method-by-Method Breakdown:
{'-'*60}"""
        
        for method_name, vote_info in result['method_votes'].items():
            status = "✅" if vote_info['detected'] else "❌"
            report += f"\n{status} {method_name}: {vote_info['detected']}"
            if vote_info.get('confidence') is not None:
                conf = vote_info['confidence']
                if isinstance(conf, float) and conf < 1.5:  # Likely a percentage
                    report += f" (confidence: {conf:.2%})"
                else:
                    report += f" (value: {conf:.3f})"
        
        report += f"\n{'='*60}\n"
        return report
