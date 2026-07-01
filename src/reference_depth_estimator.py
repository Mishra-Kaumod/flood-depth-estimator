"""
Reference Object Depth Estimator
=================================
Estimates flood depth using VISUAL REFERENCE OBJECTS — vehicles and people —
as physical scale anchors. This approach works WITHOUT a trained model:

  A car tire submerged = depth ≥ 30-35 cm
  Car bumper hidden   = depth ≥ 45 cm
  Car door handle gone = depth ≥ 90 cm
  Person knee-deep    = depth ≈ 40-50 cm
  Person waist-deep   = depth ≈ 80-100 cm

Key insight: The EfficientNet-B0 regression model learns IMPLICITLY from pixel
patterns, but a collapsed or under-trained model gives 0 cm for everything.
This module provides a physics-grounded fallback that always gives a plausible
estimate based on WHERE the waterline sits relative to known object heights.

Typical South Indian flood water is muddy brown (NOT blue), so the detector is
tuned for brown/grey water in addition to clear blue water.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─── Known reference heights (metres) ────────────────────────────────────────
# Used to label training images AND to calibrate waterline-based estimates.
REFERENCE_HEIGHTS_M = {
    # Vehicles (South India context: cars, autos, bikes common)
    "bike_tire":         0.26,   # scooter / bike tire radius
    "car_tire":          0.32,   # typical car tire radius
    "auto_bumper":       0.30,   # autorickshaw floor level
    "car_bumper":        0.45,   # car front bumper / number plate
    "car_door_sill":     0.35,   # bottom of car door opening
    "car_door_handle":   0.95,   # door handle height
    "car_hood":          1.15,   # bonnet/hood height
    "car_roof":          1.50,   # typical sedan roof
    "suv_roof":          1.75,   # SUV / van roof
    "truck_floor":       0.80,   # truck cargo floor
    # People (rough proportional guides)
    "person_ankle":      0.12,
    "person_shin":       0.28,
    "person_knee":       0.48,
    "person_thigh":      0.70,
    "person_waist":      0.92,
    "person_chest":      1.25,
    # Infrastructure
    "road_curb":         0.15,   # standard road curb height
    "speed_breaker":     0.10,
    "doorstep":          0.15,
    "manhole_rim":       0.00,   # flush with road
}

# Waterline depth lookup (for when reference objects aren't detected)
# Maps waterline position (fraction from bottom of image) → depth cm
# Calibrated for typical street-level flood photography
_WATERLINE_DEPTH_CURVE = [
    # (waterline_frac_from_bottom, depth_cm)
    (0.00,   0),   # no water
    (0.05,   5),   # ankle — barely visible
    (0.12,  12),   # ankle high
    (0.22,  22),   # shin deep
    (0.35,  35),   # knee deep
    (0.50,  52),   # thigh deep
    (0.65,  80),   # waist deep
    (0.78, 110),   # chest deep
    (0.90, 150),   # above head
]


def _interp_depth(waterline_frac: float) -> float:
    """Linearly interpolate depth from waterline position (fraction from bottom)."""
    if waterline_frac <= 0:
        return 0.0
    curve = _WATERLINE_DEPTH_CURVE
    for i in range(len(curve) - 1):
        f0, d0 = curve[i]
        f1, d1 = curve[i + 1]
        if f0 <= waterline_frac <= f1:
            t = (waterline_frac - f0) / (f1 - f0)
            return d0 + t * (d1 - d0)
    return curve[-1][1]  # beyond top = maximum depth


class ReferenceDepthEstimator:
    """
    Estimates flood depth using waterline detection and reference object analysis.

    This estimator always returns a physically plausible depth even when the
    main ML model is not trained or has collapsed.
    """

    def __init__(self):
        self._water_detector_cache = None

    # ── Public API ──────────────────────────────────────────────────────────

    def estimate(self, image_rgb: np.ndarray) -> Dict:
        """
        Estimate flood depth from an RGB numpy array (H, W, 3).

        Returns a dict with:
          depth_cm       — estimated depth in cm
          confidence     — 0-1 confidence score
          method         — which sub-method was used
          waterline_pct  — waterline height as % from bottom (0=no water, 100=top)
          water_coverage — fraction of image covered by water
          visual_cues    — list of detected reference cues
          label_guide    — suggested label for training (based on visual cues)
        """
        if not CV2_AVAILABLE:
            return self._fallback_numpy(image_rgb)

        h, w = image_rgb.shape[:2]

        # 1. Detect water region
        water_mask, water_frac = self._detect_water(image_rgb)

        if water_frac < 0.02:
            return {
                "depth_cm": 0.0, "confidence": 0.70,
                "method": "no_water_detected",
                "waterline_pct": 0.0, "water_coverage": round(water_frac, 4),
                "visual_cues": [], "label_guide": "0 cm — no water visible",
            }

        # 2. Find waterline (topmost boundary of water in the lower region)
        waterline_y, confidence = self._find_waterline(water_mask, h)
        waterline_frac = (h - waterline_y) / h   # fraction from bottom

        # 3. Detect reference objects and refine estimate
        vehicles, people = self._detect_reference_objects(image_rgb, water_mask)
        visual_cues = []
        refined_depth = None

        if vehicles:
            d, cue = self._estimate_from_vehicle(vehicles, water_mask, h, w)
            if d is not None:
                refined_depth = d
                visual_cues.append(cue)
                confidence = min(confidence + 0.15, 0.90)

        if people:
            d, cue = self._estimate_from_person(people, water_mask, h)
            if d is not None:
                if refined_depth is None:
                    refined_depth = d
                else:
                    refined_depth = (refined_depth + d) / 2  # average
                visual_cues.append(cue)
                confidence = min(confidence + 0.10, 0.90)

        # 4. Final depth: prefer reference-object estimate, fall back to waterline
        if refined_depth is not None:
            depth_cm = refined_depth * 0.6 + _interp_depth(waterline_frac) * 0.4
        else:
            depth_cm = _interp_depth(waterline_frac)
            visual_cues.append(f"waterline at {waterline_frac*100:.0f}% from bottom")

        depth_cm = max(0.0, round(depth_cm, 1))
        label_guide = self._make_label_guide(depth_cm, visual_cues)

        return {
            "depth_cm": depth_cm,
            "confidence": round(confidence, 3),
            "method": "reference_object_cv",
            "waterline_pct": round(waterline_frac * 100, 1),
            "water_coverage": round(water_frac, 4),
            "visual_cues": visual_cues,
            "label_guide": label_guide,
        }

    # ── Water detection ─────────────────────────────────────────────────────

    def _detect_water(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect water pixels.
        Handles: clear blue water, murky brown/grey Bengaluru flood water,
        reflective grey water.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_img = img.shape[0]

        # Clear blue water (ponds, clean runoff)
        blue = cv2.inRange(hsv, np.array([95, 40, 40]), np.array([140, 255, 255]))

        # Muddy brown / ochre flood water (most Bengaluru floods)
        brown = cv2.inRange(hsv, np.array([8, 25, 40]), np.array([32, 200, 210]))

        # Dark grey / reflective water (overcast, night, shade)
        grey_lower = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 45, 160]))
        # Only in lower 70% of image to exclude sky
        grey_lower[:int(h_img * 0.30)] = 0

        # Teal / green-grey turbid water
        teal = cv2.inRange(hsv, np.array([75, 20, 40]), np.array([100, 150, 200]))

        combined = cv2.bitwise_or(blue, brown)
        combined = cv2.bitwise_or(combined, grey_lower)
        combined = cv2.bitwise_or(combined, teal)

        # Morphological cleanup
        k5 = np.ones((5, 5), np.uint8)
        k11 = np.ones((11, 11), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k11, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k5, iterations=1)

        water_mask = (combined > 0).astype(np.uint8)
        water_frac = water_mask.mean()
        return water_mask, float(water_frac)

    # ── Waterline detection ─────────────────────────────────────────────────

    def _find_waterline(self, water_mask: np.ndarray, h: int) -> Tuple[int, float]:
        """
        Find the top boundary of the water region (waterline pixel row).
        Returns (waterline_y, confidence).
        """
        # Row-wise water density (0-1)
        row_density = water_mask.mean(axis=1)   # shape (H,)

        # Smooth to reduce noise
        kernel = np.ones(9) / 9
        smooth = np.convolve(row_density, kernel, mode='same')

        # Find topmost row with >20% water density (after the bottom third)
        lower_bound = int(h * 0.60)  # Look in lower 40% of image for waterline
        candidate_rows = np.where(smooth[:lower_bound] > 0.20)[0]

        if len(candidate_rows) == 0:
            # No clear waterline in lower region — expand search
            candidate_rows = np.where(smooth > 0.10)[0]
            confidence = 0.45
        else:
            confidence = 0.65

        waterline_y = int(candidate_rows.min()) if len(candidate_rows) > 0 else h

        # Higher confidence if waterline is well-defined (sharp boundary)
        if waterline_y < h:
            above = smooth[max(0, waterline_y - 10):waterline_y].mean()
            below = smooth[waterline_y:min(h, waterline_y + 10)].mean()
            sharpness = below - above
            if sharpness > 0.3:
                confidence = min(confidence + 0.15, 0.80)

        return waterline_y, confidence

    # ── Reference object detection ──────────────────────────────────────────

    def _detect_reference_objects(
        self, img: np.ndarray, water_mask: np.ndarray
    ) -> Tuple[list, list]:
        """
        Detect blobs that could be vehicles or people using contour analysis.
        Returns (vehicle_contours, person_contours).
        """
        h, w = img.shape[:2]

        # Create non-water mask (the objects we want to find)
        non_water = (1 - water_mask).astype(np.uint8) * 255

        # Focus on lower 80% of image (skip sky)
        non_water[:int(h * 0.20)] = 0

        # Edge detection on non-water region
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_nonwater = cv2.bitwise_and(gray, gray, mask=non_water)
        edges = cv2.Canny(gray_nonwater, 50, 150)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vehicles, people = [], []
        min_area = (h * w) * 0.005   # at least 0.5% of image

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            rel_height = ch / h        # height as fraction of image
            bottom_y = (y + ch) / h   # bottom edge, fraction from top

            # Vehicles: wide, low-aspect, near bottom of image
            if (0.8 < aspect < 6.0 and rel_height > 0.08
                    and bottom_y > 0.65 and cw > w * 0.10):
                vehicles.append({"rect": (x, y, cw, ch), "area": area})

            # People: taller than wide, standing in water, near bottom
            elif (0.15 < aspect < 0.9 and rel_height > 0.12
                    and bottom_y > 0.60 and ch > h * 0.12):
                people.append({"rect": (x, y, cw, ch), "area": area})

        # Sort by area descending (biggest first)
        vehicles.sort(key=lambda v: v["area"], reverse=True)
        people.sort(key=lambda p: p["area"], reverse=True)

        return vehicles[:3], people[:3]

    # ── Depth estimation from objects ───────────────────────────────────────

    def _estimate_from_vehicle(
        self, vehicles: list, water_mask: np.ndarray, h: int, w: int
    ) -> Tuple[Optional[float], str]:
        """
        Estimate depth from vehicle submersion.
        Returns (depth_cm, cue_description).
        """
        for v in vehicles:
            x, y, vw, vh = v["rect"]

            # Check how much of the vehicle's lower portion is under water
            lower_20pct = max(1, int(vh * 0.20))
            vehicle_bottom_slice = water_mask[y + vh - lower_20pct: y + vh, x: x + vw]
            bottom_water = vehicle_bottom_slice.mean()

            # Check at tire level (bottom 10% of detected vehicle region)
            tire_row = y + vh - int(vh * 0.10)
            if tire_row < h:
                tire_slice = water_mask[tire_row:y + vh, x: x + vw]
                tire_water = tire_slice.mean()
            else:
                tire_water = 0.0

            if bottom_water < 0.15:
                continue  # Vehicle not in water

            # Estimate waterline on the vehicle itself
            vehicle_mask = water_mask[y:y + vh, x:x + vw]
            row_water = vehicle_mask.mean(axis=1)
            wet_rows = np.where(row_water > 0.25)[0]

            if len(wet_rows) == 0:
                continue

            topmost_wet = wet_rows.min()
            waterline_on_vehicle = topmost_wet / vh   # 0=top of vehicle, 1=bottom

            # Map waterline position to depth
            # 0.95-1.0 → tire only → 30-40cm
            # 0.80-0.95 → bumper → 40-55cm
            # 0.60-0.80 → door sill/lower door → 55-75cm
            # 0.40-0.60 → door handle → 75-100cm
            # 0.20-0.40 → hood → 100-130cm
            # <0.20     → roof → 130-160cm
            if waterline_on_vehicle > 0.90:
                depth_cm = 32
                cue = "vehicle tire submerged (depth ~32 cm)"
            elif waterline_on_vehicle > 0.75:
                depth_cm = 48
                cue = "vehicle bumper submerged (depth ~48 cm)"
            elif waterline_on_vehicle > 0.55:
                depth_cm = 68
                cue = "vehicle lower door submerged (depth ~68 cm)"
            elif waterline_on_vehicle > 0.35:
                depth_cm = 95
                cue = "vehicle door handle submerged (depth ~95 cm)"
            elif waterline_on_vehicle > 0.15:
                depth_cm = 120
                cue = "vehicle hood submerged (depth ~120 cm)"
            else:
                depth_cm = 148
                cue = "vehicle nearly fully submerged (depth ~148 cm)"

            return float(depth_cm), cue

        return None, ""

    def _estimate_from_person(
        self, people: list, water_mask: np.ndarray, h: int
    ) -> Tuple[Optional[float], str]:
        """
        Estimate depth from person submersion level.
        Returns (depth_cm, cue_description).
        """
        for p in people:
            x, y, pw, ph = p["rect"]

            person_mask = water_mask[y:y + ph, x:x + pw]
            row_water = person_mask.mean(axis=1)
            wet_rows = np.where(row_water > 0.25)[0]

            if len(wet_rows) == 0:
                continue

            topmost_wet = wet_rows.min()
            waterline_on_person = topmost_wet / ph  # 0=top, 1=bottom

            # Map to depth based on body proportions
            if waterline_on_person > 0.90:
                depth_cm = 12
                cue = "person ankle-deep (depth ~12 cm)"
            elif waterline_on_person > 0.75:
                depth_cm = 28
                cue = "person shin-deep (depth ~28 cm)"
            elif waterline_on_person > 0.55:
                depth_cm = 48
                cue = "person knee-deep (depth ~48 cm)"
            elif waterline_on_person > 0.40:
                depth_cm = 72
                cue = "person thigh-deep (depth ~72 cm)"
            elif waterline_on_person > 0.25:
                depth_cm = 92
                cue = "person waist-deep (depth ~92 cm)"
            else:
                depth_cm = 125
                cue = "person chest-deep (depth ~125 cm)"

            return float(depth_cm), cue

        return None, ""

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_label_guide(depth_cm: float, cues: list) -> str:
        """Human-readable label suggestion for training."""
        if depth_cm < 5:
            severity = "No flood / dry road"
        elif depth_cm < 15:
            severity = "Very shallow — ankle level"
        elif depth_cm < 30:
            severity = "Shallow — shin level"
        elif depth_cm < 55:
            severity = "Moderate — knee level"
        elif depth_cm < 80:
            severity = "Deep — thigh / door sill level"
        elif depth_cm < 110:
            severity = "Very deep — waist / door handle level"
        else:
            severity = "Extreme — hood / roof level"

        cue_str = ("; ".join(cues[:2])) if cues else "waterline position only"
        return f"{depth_cm:.0f} cm — {severity} (based on: {cue_str})"

    def _fallback_numpy(self, img: np.ndarray) -> Dict:
        """Pure-numpy fallback when OpenCV is not available."""
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        h = img.shape[0]

        # Simple blue / brown water detection
        blue = (b.astype(int) - r.astype(int) > 20) & (b > 60)
        brown = (r.astype(int) - b.astype(int) > 20) & (r > 80)
        water = (blue | brown).astype(float)
        water_frac = water.mean()

        if water_frac < 0.03:
            return {"depth_cm": 0.0, "confidence": 0.40, "method": "numpy_fallback",
                    "waterline_pct": 0.0, "water_coverage": 0.0,
                    "visual_cues": [], "label_guide": "0 cm — no water detected"}

        water_rows = np.where(water.mean(axis=1) > 0.10)[0]
        waterline_y = water_rows.min() if len(water_rows) > 0 else h
        waterline_frac = (h - waterline_y) / h
        depth_cm = _interp_depth(waterline_frac)

        return {
            "depth_cm": round(depth_cm, 1),
            "confidence": round(min(0.35 + water_frac * 0.40, 0.65), 3),
            "method": "numpy_fallback",
            "waterline_pct": round(waterline_frac * 100, 1),
            "water_coverage": round(water_frac, 4),
            "visual_cues": [f"waterline at {waterline_frac*100:.0f}% from bottom"],
            "label_guide": f"{depth_cm:.0f} cm (numpy estimate)",
        }


# ─── Convenience function ────────────────────────────────────────────────────

def label_from_image(image_path: str) -> Dict:
    """
    Quick utility: run reference estimator on a single image file.
    Useful for auto-labeling training images.

    Usage:
        from src.reference_depth_estimator import label_from_image
        result = label_from_image('flood_photo.jpg')
        print(f"Suggested label: {result['label_guide']}")
    """
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert('RGB')
    arr = np.array(img)
    estimator = ReferenceDepthEstimator()
    result = estimator.estimate(arr)
    result["image_path"] = image_path
    return result
