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


class WaterDetectionAnalyzer:
    """Advanced water surface detection using multiple computer vision techniques."""

    def __init__(self):
        self.min_water_area_pct = 0.05
        self.method_weights = {
            "validated_water_region": 0.35,
            "rgb_color_analysis": 0.20,
            "edge_detection": 0.10,
            "contrast_analysis": 0.10,
            "horizontal_line_detection": 0.15,
            "depth_discontinuity": 0.05,
            "optical_flow_ripples": 0.05,
        }
        self.water_threshold = 0.45

    def detect_water_surface(self, image, depth_map=None):
        """
        Comprehensive water detection using multiple methods.

        Args:
            image: BGR image
            depth_map: Optional normalized depth map (0-1)

        Returns:
            dict with detection results
        """
        if image is None or image.size == 0:
            return {
                "water_detected": False,
                "confidence": 0.0,
                "water_percentage": 0.0,
                "method_votes": {},
                "details": {},
                "water_mask": np.zeros((0, 0), dtype=np.uint8),
            }

        h, w = image.shape[:2]
        road_score = self._road_visibility_score(image)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        validated_region = self._detect_validated_water_region(rgb_image)
        results = {
            "validated_water_region": validated_region,
            "rgb_color_analysis": self._detect_water_by_color(image),
            "edge_detection": self._detect_water_edges(image),
            "contrast_analysis": self._detect_water_contrast(image),
            "horizontal_line_detection": self._detect_water_surface_line(image),
            "depth_discontinuity": self._detect_depth_discontinuity(depth_map) if depth_map is not None else None,
            "optical_flow_ripples": self._detect_ripple_patterns(image),
        }

        consensus = self._aggregate_detections(results, h, w)
        consensus["road_visibility"] = road_score
        return {
            "water_detected": consensus["water_detected"],
            "confidence": consensus["confidence"],
            "water_percentage": consensus["water_percentage"],
            "method_votes": consensus["method_votes"],
            "details": results,
            "water_mask": consensus["water_mask"],
        }

    def _detect_validated_water_region(self, image_rgb):
        mask, water_pct, confidence, flags = self._detect_region_mask(image_rgb)
        water_detected = bool(mask.size > 0 and water_pct >= self.min_water_area_pct and confidence >= 0.25)
        return {
            "water_detected": water_detected,
            "percentage": float(water_pct),
            "confidence": float(confidence),
            "flags": flags,
            "mask": mask,
            "method": "Validated Water Region",
        }

    def _detect_region_mask(self, image_rgb):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask = np.maximum(mask, self._detect_hsv(image_rgb))
        mask = np.maximum(mask, self._detect_rgb(image_rgb))
        mask = np.maximum(mask, self._detect_flatness(image_rgb))
        mask = self._morphological_cleanup(mask)

        hard_zero = int(h * 0.30)
        ramp_end = int(h * 0.45)
        mask[:hard_zero, :] = 0
        if ramp_end > hard_zero:
            ramp = np.linspace(0.0, 1.0, ramp_end - hard_zero, dtype=np.float32)
            mask[hard_zero:ramp_end, :] = np.clip(
                mask[hard_zero:ramp_end, :].astype(np.float32) * ramp[:, np.newaxis],
                0,
                255,
            ).astype(np.uint8)

        water_pixels = int(np.count_nonzero(mask > 0))
        if water_pixels < 50:
            return mask, float(water_pixels) / max(1, h * w), 0.05, []

        mask, score, flags = self._validate_and_refine(mask, image_rgb)
        water_pct = float(np.count_nonzero(mask > 0)) / max(1, h * w)
        return mask, water_pct, float(score), flags

    def _detect_hsv(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        clear = ((h_ >= 85) & (h_ <= 140) & (s_ >= 40) & (v_ >= 40)).astype(np.uint8) * 255
        muddy_h = (h_ <= 25) | (h_ >= 160)
        muddy = (muddy_h & (s_ >= 15) & (s_ <= 190) & (v_ >= 45) & (v_ <= 225)).astype(np.uint8) * 255
        grey = ((s_ < 30) & (v_ >= 50) & (v_ <= 160)).astype(np.uint8) * 255
        return np.maximum(np.maximum(clear, muddy), grey)

    def _detect_rgb(self, image):
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)
        bright = (r + g + b) / 3.0

        blue = ((b > g) & (b > r) & ((b - r) > 15) & (b > 30)).astype(np.uint8) * 255
        brown = (
            (r > b)
            & (g > b)
            & (r > 40)
            & (r < 215)
            & ((r - b) > 10)
            & ((r - b) < 110)
            & (bright < 205)
            & (bright > 28)
        ).astype(np.uint8) * 255
        dark = (
            (bright > 35)
            & (bright < 110)
            & (np.abs(r - g) < 25)
            & (np.abs(g - b) < 25)
            & (np.abs(r - b) < 25)
        ).astype(np.uint8) * 255
        return np.maximum(np.maximum(blue, brown), dark)

    def _detect_flatness(self, image):
        """Low-texture regions in the bottom half are likely water surface."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h = gray.shape[0]
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        tex = cv2.GaussianBlur(lap, (11, 11), 0)
        global_tex_std = float(np.std(tex))
        if global_tex_std < 1.5:
            return np.zeros(gray.shape, dtype=np.uint8)

        thresh = float(np.percentile(tex, 30))
        min_abs_thresh = max(thresh, 1.0)
        flat = (tex <= min_abs_thresh).astype(np.float32)
        flat[: int(h * 0.40), :] = 0.0
        return (flat * 255).astype(np.uint8)

    def _morphological_cleanup(self, mask, kernel_size=7):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _validate_and_refine(self, mask, image_rgb):
        h, _ = mask.shape
        water_pixels = int(np.sum(mask > 0))
        flags = []
        score = 1.0

        if water_pixels < 50:
            return mask, 1.0, flags

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        abs_lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

        water_tex_mean = float(np.mean(abs_lap[mask > 0]))
        if water_tex_mean > 28:
            flags.append("HIGH_TEXTURE")
            score -= 0.40
        elif water_tex_mean > 18:
            flags.append("MODERATE_TEXTURE")
            score -= 0.20

        tex_smooth = cv2.GaussianBlur(abs_lap, (9, 9), 0)
        water_tex_vals = tex_smooth[mask > 0]
        tex_thresh = float(np.percentile(water_tex_vals, 65))
        refined = np.where((mask > 0) & (tex_smooth <= tex_thresh), 255, 0).astype(np.uint8)
        refined = self._morphological_cleanup(refined, kernel_size=5)
        if int(np.sum(refined > 0)) < water_pixels * 0.20:
            refined = mask
        else:
            mask = refined

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = int(np.max(areas))
            total_a = int(np.sum(areas))
            conn_ratio = largest / total_a if total_a > 0 else 1.0
            if conn_ratio < 0.35:
                flags.append("FRAGMENTED")
                score -= 0.25
            elif conn_ratio < 0.55:
                flags.append("SCATTERED")
                score -= 0.10

        ys, _ = np.where(mask > 0)
        if len(ys) > 0:
            com_y = float(np.mean(ys)) / h
            if com_y < 0.38:
                flags.append("WATER_TOO_HIGH")
                score -= 0.30
            elif com_y < 0.50:
                flags.append("WATER_UPPER_MIDFRAME")
                score -= 0.10

        water_px_rgb = image_rgb[mask > 0]
        if water_px_rgb.shape[0] > 50:
            color_std = float(np.mean(np.std(water_px_rgb.astype(float), axis=0)))
            if color_std > 50:
                flags.append("COLOR_DIVERSE")
                score -= 0.15

        score = float(np.clip(score, 0.05, 1.0))
        return mask, score, flags

    def _detect_water_by_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        blue_mask = cv2.inRange(hsv, np.array([90, 40, 50]), np.array([130, 255, 255]))
        reflective_mask = cv2.inRange(s, 0, 60) & cv2.inRange(v, 80, 180)
        water_color_mask = cv2.bitwise_or(blue_mask, reflective_mask)

        height = image.shape[0]
        water_color_mask[: int(height * 0.4), :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_color_mask = cv2.morphologyEx(water_color_mask, cv2.MORPH_OPEN, kernel)
        water_color_mask = cv2.morphologyEx(water_color_mask, cv2.MORPH_CLOSE, kernel)

        water_pct = np.count_nonzero(water_color_mask) / water_color_mask.size
        return {
            "water_detected": water_pct > 0.03,
            "percentage": float(water_pct),
            "mask": water_color_mask,
            "method": "Color-based (HSV)",
        }

    def _detect_water_edges(self, image):
        """Water surface edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 20:
                    horizontal_lines += 1

        edge_strength = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
        return {
            "water_detected": horizontal_lines >= 3,
            "horizontal_lines": int(horizontal_lines),
            "edge_strength": float(edge_strength),
            "mask": edges,
            "method": "Horizontal Edge Lines",
        }

    def _detect_water_contrast(self, image):
        """Contrast pattern analysis for water."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, _ = gray.shape

        band_height = h // 5
        contrast_values = []

        for i in range(5):
            band = gray[i * band_height : (i + 1) * band_height, :]
            contrast = cv2.Laplacian(band, cv2.CV_64F)
            contrast_values.append(np.std(contrast))

        pattern_match = contrast_values[0] > contrast_values[1] and contrast_values[1] < contrast_values[2]

        avg_contrast = np.mean(contrast_values)
        contrast_map = cv2.Laplacian(gray, cv2.CV_64F)
        contrast_map = cv2.normalize(contrast_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, contrast_mask = cv2.threshold(contrast_map, 30, 255, cv2.THRESH_BINARY_INV)

        return {
            "water_detected": pattern_match,
            "avg_contrast": float(avg_contrast),
            "pattern_match": pattern_match,
            "mask": contrast_mask,
            "method": "Contrast Pattern Analysis",
        }

    def _detect_water_surface_line(self, image):
        """Horizontal surface line detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, _ = gray.shape

        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobely = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        horizontal_strength = sobely.astype(np.float32)
        col_prominence = np.mean(horizontal_strength, axis=1)

        lower_half_prominence = col_prominence[h // 2 :]
        prominent_rows = np.where(lower_half_prominence > np.percentile(lower_half_prominence, 75))[0]

        continuous_band = len(prominent_rows) > 20
        max_prominence = float(np.max(col_prominence))

        threshold = np.percentile(horizontal_strength, 80)
        surface_mask = (horizontal_strength > threshold).astype(np.uint8) * 255
        return {
            "water_detected": continuous_band and max_prominence > 50,
            "max_prominence": max_prominence,
            "continuous_band": continuous_band,
            "mask": surface_mask,
            "method": "Horizontal Surface Line",
        }

    def _detect_depth_discontinuity(self, depth_map):
        """Depth discontinuity detection."""
        if depth_map is None:
            return None

        grad_y = np.gradient(depth_map, axis=0)
        grad_x = np.gradient(depth_map, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        boundary_mask = (grad_mag > np.percentile(grad_mag, 85)).astype(np.uint8) * 255
        horizontal_discontinuity = np.abs(grad_y) > np.percentile(np.abs(grad_y), 80)

        h, w = depth_map.shape
        discontinuity_pct = np.count_nonzero(horizontal_discontinuity) / (h * w)
        return {
            "water_detected": discontinuity_pct > 0.10,
            "discontinuity_percentage": float(discontinuity_pct),
            "mask": boundary_mask,
            "method": "Depth Discontinuity",
        }

    def _detect_ripple_patterns(self, image):
        """Ripple/motion pattern detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.blur(gray, (5, 5))
        sqmean = cv2.blur(gray * gray, (5, 5))
        local_var = sqmean - mean**2

        ripple_mask = (
            (local_var > np.percentile(local_var, 30)) & (local_var < np.percentile(local_var, 90))
        ).astype(np.uint8) * 255
        ripple_pct = np.count_nonzero(ripple_mask) / ripple_mask.size
        return {
            "water_detected": ripple_pct > 0.15,
            "ripple_percentage": float(ripple_pct),
            "mask": ripple_mask,
            "method": "Ripple Pattern Detection",
        }

    def _aggregate_detections(self, results, h, w):
        """Aggregate results from all methods using weighted voting."""
        weighted_score = 0.0
        total_weight = 0.0
        method_votes = {}
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for method_name, result in results.items():
            if result is None:
                continue

            weight = self.method_weights.get(method_name, 0.1)
            total_weight += weight

            detected = bool(result.get("water_detected", False))
            if detected:
                weighted_score += weight

            method_votes[method_name] = {
                "detected": detected,
                "weight": weight,
                "confidence": float(result.get("confidence", result.get("percentage", 0.0))),
            }

            if "mask" in result and result["mask"] is not None:
                combined_mask = cv2.bitwise_or(combined_mask, result["mask"].astype(np.uint8))

        confidence = weighted_score / total_weight if total_weight > 0 else 0.0
        validated_region = results.get("validated_water_region") or {}
        validated_detected = bool(validated_region.get("water_detected", False))
        validated_pct = float(validated_region.get("percentage", 0.0))

        water_detected = confidence >= self.water_threshold or (validated_detected and validated_pct >= self.min_water_area_pct)
        water_pct = np.count_nonzero(combined_mask) / max(1, h * w) if h * w > 0 else 0.0
        if validated_detected:
            water_pct = max(water_pct, validated_pct)

        return {
            "water_detected": water_detected,
            "confidence": float(confidence),
            "water_percentage": float(water_pct),
            "method_votes": method_votes,
            "water_mask": combined_mask,
        }

    def _road_visibility_score(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10)
        visible_lines = len(lines) if lines is not None else 0
        return min(1.0, visible_lines / 20)