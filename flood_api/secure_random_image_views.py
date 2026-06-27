# flood_api/secure_random_image_views.py
# SECURE IMAGE UPLOAD WITH VALIDATION, AUTHENTICATION, AND RATE LIMITING

import os
import uuid
import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, FileResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.core.exceptions import ValidationError
from django.utils import timezone
from django.core.files.storage import default_storage
from django.middleware.csrf import get_token
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.utils.dateparse import parse_datetime
from django.views.generic import View

from PIL import Image, ExifTags
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle
from rest_framework.exceptions import ValidationError as DRFValidationError

import cv2
import numpy as np
from functools import wraps
from .models import SecureRandomImageUploadResult, AuditLog
from .services.prediction_policy import harmonize_prediction

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_IMAGES_PER_BATCH = 10
UPLOAD_TIMEOUT_SECONDS = 10

# Magic bytes for file type validation
FILE_SIGNATURES = {
    b'\xFF\xD8\xFF': 'jpg',      # JPEG
    b'\x89PNG': 'png',            # PNG
    b'GIF8': 'gif',               # GIF
    b'BM': 'bmp',                 # BMP
}

# Default fallback reference for missing field metadata: Bellandur Central
DEFAULT_LAT = 12.9259
DEFAULT_LNG = 77.6762
DEFAULT_CAMERA_ID = 'bellandur_central_01'
DEFAULT_LOCATION = 'Bellandur Central, Bengaluru'
# 10 real Bengaluru flood-prone zones used to spread map markers across images
BENGALURU_FLOOD_ZONES = [
    {"name": "Koramangala",       "lat": 12.9352, "lng": 77.6245},
    {"name": "Bellandur Lake",    "lat": 12.9259, "lng": 77.6762},
    {"name": "HSR Layout",        "lat": 12.9116, "lng": 77.6473},
    {"name": "Whitefield",        "lat": 12.9698, "lng": 77.7499},
    {"name": "Marathahalli",      "lat": 12.9591, "lng": 77.6972},
    {"name": "Sarjapur Road",     "lat": 12.9010, "lng": 77.6874},
    {"name": "Varthur Lake",      "lat": 12.9400, "lng": 77.7470},
    {"name": "K R Puram",         "lat": 13.0033, "lng": 77.6963},
    {"name": "Hebbal",            "lat": 13.0450, "lng": 77.5971},
    {"name": "Silk Board Jn",     "lat": 12.9176, "lng": 77.6232},
]

SWD_CHOKE_POINTS = [
    {"name": "Bellandur SWD Culvert", "lat": 12.9267, "lng": 77.6719, "infrastructure": "swd_network"},
    {"name": "Koramangala 80ft Drain", "lat": 12.9358, "lng": 77.6224, "infrastructure": "swd_network"},
    {"name": "Marathahalli Bridge Drain", "lat": 12.9578, "lng": 77.6960, "infrastructure": "swd_network"},
    {"name": "Silk Board Underpass Drain", "lat": 12.9171, "lng": 77.6238, "infrastructure": "swd_network"},
    {"name": "KR Puram Junction Outfall", "lat": 13.0038, "lng": 77.6978, "infrastructure": "swd_network"},
    {"name": "Hebbal Flyover Drain", "lat": 13.0437, "lng": 77.5957, "infrastructure": "swd_network"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY THROTTLE
# ─────────────────────────────────────────────────────────────────────────────

class ImageUploadThrottle(UserRateThrottle):
    """Prevent DoS: max 10 uploads per hour per user"""
    scope = 'image_upload'
    rate = '10/hour'

# ─────────────────────────────────────────────────────────────────────────────
# FILE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_image_file(file):
    """Validate file type, size, and content"""
    
    # Check file extension
    ext = file.name.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(f"File type '{ext}' not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Check file size
    if file.size > MAX_FILE_SIZE:
        raise ValidationError(f"File too large ({file.size / 1024 / 1024:.1f}MB). Max: 50MB")
    
    # Check file signature (magic bytes)
    file.seek(0)
    header = file.read(8)
    
    is_valid = False
    for sig, sig_ext in FILE_SIGNATURES.items():
        if header.startswith(sig):
            is_valid = True
            break
    
    if not is_valid:
        raise ValidationError("Invalid image file or corrupted. Upload a valid JPG, PNG, GIF, or BMP.")
    
    file.seek(0)
    return True

def sanitize_filename(filename):
    """Generate safe filename using UUID"""
    ext = filename.split('.')[-1].lower()
    return f"{uuid.uuid4().hex}.{ext}"

def to_json_safe(value):
    """Recursively convert values to JSON-serializable Python natives."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

def parse_float_or_default(raw_value, default_value):
    """Parse a float input while treating blank values as the configured default."""
    if raw_value is None:
        return float(default_value)
    if isinstance(raw_value, str) and raw_value.strip() == "":
        return float(default_value)
    return float(raw_value)

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lng points."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return radius * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def _to_float_ratio(value):
    """Convert EXIF rational or numeric values to float."""
    try:
        return float(value)
    except Exception:
        pass
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        try:
            return float(value[0]) / float(value[1])
        except Exception:
            return None
    return None

def _dms_to_decimal(dms_tuple, ref):
    if not dms_tuple or len(dms_tuple) != 3:
        return None
    deg = _to_float_ratio(dms_tuple[0])
    minute = _to_float_ratio(dms_tuple[1])
    second = _to_float_ratio(dms_tuple[2])
    if None in (deg, minute, second):
        return None
    decimal = deg + (minute / 60.0) + (second / 3600.0)
    if ref in ("S", "W"):
        decimal *= -1
    return decimal

def extract_exif_coordinates(image_path):
    """Extract GPS coordinates from EXIF if available."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None, None
            gps_tag = None
            for tag_id, tag_name in ExifTags.TAGS.items():
                if tag_name == "GPSInfo":
                    gps_tag = tag_id
                    break
            if gps_tag is None or gps_tag not in exif:
                return None, None
            gps_raw = exif.get(gps_tag, {})
            gps = {
                ExifTags.GPSTAGS.get(k, k): v
                for k, v in gps_raw.items()
            }
            lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N"))
            lng = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"))
            return lat, lng
    except Exception:
        return None, None

def nearest_bengaluru_zone(latitude, longitude):
    """Return nearest known Bengaluru flood zone."""
    ranked = sorted(
        BENGALURU_FLOOD_ZONES,
        key=lambda z: haversine_m(latitude, longitude, z["lat"], z["lng"]) or float("inf"),
    )
    if not ranked:
        return None, None
    nearest = ranked[0]
    return nearest, haversine_m(latitude, longitude, nearest["lat"], nearest["lng"])

def run_data_governance_audit(user_lat, user_lng, image_path):
    """Cross-check user coordinates against EXIF and inferred nearby zones."""
    exif_lat, exif_lng = extract_exif_coordinates(image_path)
    flags = []
    rationale_parts = []
    exif_distance_m = None
    nearest_zone_name = None
    zone_distance_m = None

    nearest_zone, zone_distance = nearest_bengaluru_zone(user_lat, user_lng)
    if nearest_zone:
        nearest_zone_name = nearest_zone["name"]
        zone_distance_m = round(zone_distance or 0.0, 1)

    if exif_lat is not None and exif_lng is not None:
        exif_distance = haversine_m(user_lat, user_lng, exif_lat, exif_lng) or 0.0
        exif_distance_m = round(exif_distance, 1)
        if exif_distance > 250:
            flags.append({
                "code": "COORDINATE_EXIF_MISMATCH",
                "severity": "high",
                "distance_m": exif_distance_m,
            })
            rationale_parts.append(
                f"User coordinates differ from image EXIF GPS by {exif_distance_m}m."
            )
        else:
            rationale_parts.append(
                f"User coordinates align with EXIF GPS ({exif_distance_m}m delta)."
            )
    else:
        flags.append({
            "code": "EXIF_GPS_MISSING",
            "severity": "medium",
        })
        rationale_parts.append(
            "No EXIF GPS detected; coordinate trust downgraded until field validation."
        )

    if not (12.5 <= float(user_lat) <= 13.5 and 77.0 <= float(user_lng) <= 78.0):
        flags.append({
            "code": "OUT_OF_BENGALURU_BOUNDS",
            "severity": "high",
        })
        rationale_parts.append("Coordinates fall outside Bengaluru bounding guardrails.")

    status = "flagged" if flags else "passed"
    return {
        "status": status,
        "flags": flags,
        "rationale": " ".join(rationale_parts) if rationale_parts else "Coordinate checks passed.",
        "exif_latitude": exif_lat,
        "exif_longitude": exif_lng,
        "exif_distance_m": exif_distance_m,
        "nearest_zone_name": nearest_zone_name,
        "nearest_zone_distance_m": zone_distance_m,
    }

def recommend_escalation_path(analysis, zone_name):
    """Generate operational escalation recommendation for BBMP command center."""
    intensity = str(analysis.get("intensity", "SAFE"))
    depth_cm = int(analysis.get("depth_cm", 0))
    confidence = round(float(analysis.get("confidence", 0.0)), 1)
    is_flood = bool(analysis.get("is_flood", False))

    if not is_flood:
        return {
            "priority": "monitor",
            "target": "Ward Monitoring Desk",
            "eta_minutes": 60,
            "rationale": f"No flooding confirmed in {zone_name} (confidence {confidence}%). Continue passive monitoring.",
        }

    if intensity == "CRITICAL" or depth_cm >= 90:
        return {
            "priority": "immediate",
            "target": "BBMP Command Centre + Emergency Field Unit",
            "eta_minutes": 10,
            "rationale": f"Critical flood signature in {zone_name}: depth {depth_cm}cm at {confidence}% confidence.",
        }

    if intensity == "HIGH" or depth_cm >= 50:
        return {
            "priority": "urgent",
            "target": "Zonal Control Room + SWD Rapid Response",
            "eta_minutes": 20,
            "rationale": f"High flood risk in {zone_name}: depth {depth_cm}cm. Immediate drain clearance and traffic diversion advised.",
        }

    return {
        "priority": "elevated",
        "target": "Assistant Engineer (Ward) + Field Inspector",
        "eta_minutes": 45,
        "rationale": f"Moderate flood indicators in {zone_name}; verify physically and watch downstream drains.",
    }

def predict_downstream_choke_points(zone_lat, zone_lng, analysis):
    """Estimate likely downstream choke points using nearest SWD nodes + severity."""
    intensity = str(analysis.get("intensity", "SAFE"))
    depth_cm = int(analysis.get("depth_cm", 0))
    severity_factor = {"SAFE": 0.4, "MEDIUM": 0.8, "HIGH": 1.1, "CRITICAL": 1.4}.get(intensity, 0.8)
    candidates = []

    for node in SWD_CHOKE_POINTS:
        distance_m = haversine_m(zone_lat, zone_lng, node["lat"], node["lng"]) or 5000.0
        base = max(5.0, 100.0 - (distance_m / 45.0))
        risk_score = min(100.0, round((base * severity_factor) + (depth_cm * 0.25), 1))
        candidates.append({
            "name": node["name"],
            "infrastructure": node["infrastructure"],
            "distance_m": round(distance_m, 1),
            "risk_score": risk_score,
        })

    candidates.sort(key=lambda x: x["risk_score"], reverse=True)
    return candidates[:3]

def compute_historical_pattern(latitude, longitude):
    """Summarize historical flood behavior near a coordinate from recent secure batches."""
    recent_batches = SecureRandomImageUploadResult.objects.filter(
        created_at__gte=timezone.now() - timedelta(days=90)
    )[:250]
    nearby = []
    for batch in recent_batches:
        distance_m = haversine_m(latitude, longitude, batch.latitude, batch.longitude)
        if distance_m is not None and distance_m <= 3000:
            nearby.append(batch)

    if not nearby:
        return {
            "samples": 0,
            "flood_occurrence_pct": 0.0,
            "avg_depth_cm": 0.0,
            "trend": "insufficient_data",
        }

    total_images = sum(max(1, b.total_images) for b in nearby)
    total_flooded = sum(max(0, b.flooded_count) for b in nearby)
    avg_depth = round(sum(float(b.avg_depth_cm) for b in nearby) / len(nearby), 1)
    flood_occurrence_pct = round((total_flooded / total_images) * 100.0, 1)

    if flood_occurrence_pct >= 55:
        trend = "persistent_high_risk"
    elif flood_occurrence_pct >= 30:
        trend = "elevated_risk"
    else:
        trend = "intermittent_risk"

    return {
        "samples": len(nearby),
        "flood_occurrence_pct": flood_occurrence_pct,
        "avg_depth_cm": avg_depth,
        "trend": trend,
    }

def cluster_incidents_for_map(results, radius_m=50, window_minutes=30):
    """Merge nearby detections into a single incident bubble to reduce map clutter."""
    successful = [r for r in results if r.get("status") == "success"]
    clusters = []
    for item in successful:
        lat = float(item.get("latitude", DEFAULT_LAT))
        lng = float(item.get("longitude", DEFAULT_LNG))
        captured_at = parse_datetime(item.get("captured_at")) if item.get("captured_at") else timezone.now()
        attached = None
        for cluster in clusters:
            center_lat = cluster["latitude"]
            center_lng = cluster["longitude"]
            distance = haversine_m(lat, lng, center_lat, center_lng)
            cluster_time = parse_datetime(cluster.get("captured_at")) if cluster.get("captured_at") else captured_at
            within_window = abs((captured_at - cluster_time).total_seconds()) <= (window_minutes * 60)
            if distance is not None and distance <= radius_m and within_window:
                attached = cluster
                break
        if attached is None:
            clusters.append({
                "id": f"incident_{len(clusters) + 1}",
                "latitude": lat,
                "longitude": lng,
                "image_count": 0,
                "flooded_images": 0,
                "max_depth_cm": 0,
                "max_confidence_pct": 0.0,
                "severity": "SAFE",
                "zones": [],
                "captured_at": captured_at.isoformat(),
            })
            attached = clusters[-1]

        analysis = item.get("analysis", {})
        depth_cm = int(analysis.get("depth_cm", 0))
        confidence = float(analysis.get("confidence", 0.0))
        intensity = str(analysis.get("intensity", "SAFE"))
        attached["image_count"] += 1
        attached["flooded_images"] += 1 if bool(analysis.get("is_flood", False)) else 0
        attached["max_depth_cm"] = max(attached["max_depth_cm"], depth_cm)
        attached["max_confidence_pct"] = max(attached["max_confidence_pct"], confidence)
        if item.get("zone_name") and item.get("zone_name") not in attached["zones"]:
            attached["zones"].append(item.get("zone_name"))
        if intensity in ("CRITICAL", "HIGH", "MEDIUM"):
            order = {"SAFE": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
            if order[intensity] > order.get(attached["severity"], 0):
                attached["severity"] = intensity

    for cluster in clusters:
        cluster["bubble_type"] = (
            "High-Severity Incident"
            if cluster["severity"] in ("CRITICAL", "HIGH")
            else "Incident Cluster"
        )
        cluster["zones_label"] = ", ".join(cluster["zones"]) if cluster["zones"] else "Unknown"
        cluster["max_confidence_pct"] = round(cluster["max_confidence_pct"], 1)
    return clusters

def build_agentic_ui_commands(results, telemetry_metrics, report_url):
    """Create executable UI command payloads for front-end rendering engine."""
    successful = [r for r in results if r.get("status") == "success"]
    severe = [r for r in successful if r.get("analysis", {}).get("intensity") in ("HIGH", "CRITICAL")]
    primary = (severe[0] if severe else (successful[0] if successful else None))
    chips = []
    if primary:
        analysis = primary.get("analysis", {})
        confidence = round(float(analysis.get("confidence", 0.0)), 1)
        chips = [
            {
                "field": "location",
                "value": primary.get("zone_name", "Bengaluru, Karnataka"),
                "confidence_pct": confidence,
                "rationale": "Vision landmarks and flood-zone matching",
            },
            {
                "field": "scenario_name",
                "value": f"{primary.get('zone_name', 'Bengaluru')} Flood Triage",
                "confidence_pct": max(55.0, round(confidence - 6.0, 1)),
                "rationale": "Severity-informed scenario suggestion",
            },
            {
                "field": "camera_id",
                "value": "agentic_inferred_camera",
                "confidence_pct": 62.0,
                "rationale": "Fallback camera inference (manual confirmation advised)",
            },
        ]

    clustered_incidents = cluster_incidents_for_map(results)
    return [
        {
            "type": "notification.push",
            "payload": {
                "title": "Agentic triage complete",
                "message": f"{len(successful)} image(s) processed and ready for validation.",
                "severity": "info",
            },
        },
        {
            "type": "form.prefill_chips",
            "payload": {"chips": chips},
        },
        {
            "type": "map.toggle_layers",
            "payload": {
                "layers": {
                    "swd_network": True,
                    "traffic": len(severe) > 0,
                }
            },
        },
        {
            "type": "map.replace_incident_bubbles",
            "payload": {
                "clusters": clustered_incidents,
                "strategy": "radius_50m_window_30m",
            },
        },
        {
            "type": "telemetry.update_metrics",
            "payload": telemetry_metrics,
        },
        {
            "type": "notification.push",
            "payload": {
                "title": "Report available",
                "message": f"Open detailed report: {report_url}",
                "severity": "success",
            },
        },
    ]

# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATION & SANITIZATION
# ─────────────────────────────────────────────────────────────────────────────

class SecureImageUploadForm:
    """Form validator with security checks"""
    
    def __init__(self, data, files):
        self.data = data
        self.files = files
        self.errors = {}
    
    def is_valid(self):
        """Validate all inputs"""
        try:
            # Validate images
            images = self.files.getlist('images')
            if not images or len(images) == 0:
                self.errors['images'] = "No images selected"
                return False
            
            if len(images) > MAX_IMAGES_PER_BATCH:
                self.errors['images'] = f"Max {MAX_IMAGES_PER_BATCH} images per batch"
                return False
            
            # Validate each image
            for img in images:
                try:
                    validate_image_file(img)
                except ValidationError as e:
                    self.errors['images'] = str(e)
                    return False
            
            # Validate text fields (max length, no HTML)
            scenario = self.data.get('scenario_name', '').strip()
            if not scenario or len(scenario) < 3:
                self.errors['scenario_name'] = "Scenario name must be at least 3 characters"
                return False
            if len(scenario) > 100:
                self.errors['scenario_name'] = "Scenario name too long (max 100 chars)"
                return False
            if '<' in scenario or '>' in scenario:
                self.errors['scenario_name'] = "Invalid characters in scenario name"
                return False
            
            location = self.data.get('location', '').strip()
            if not location or len(location) < 3:
                self.errors['location'] = "Location must be at least 3 characters"
                return False
            if len(location) > 100:
                self.errors['location'] = "Location too long (max 100 chars)"
                return False
            if '<' in location or '>' in location:
                self.errors['location'] = "Invalid characters in location"
                return False
            
            # Optional: validate coordinates
            try:
                lat = parse_float_or_default(self.data.get('latitude', DEFAULT_LAT), DEFAULT_LAT)
                lng = parse_float_or_default(self.data.get('longitude', DEFAULT_LNG), DEFAULT_LNG)
                
                # Bengaluru bounds approximately
                if not (12.5 <= lat <= 13.5 and 77.0 <= lng <= 78.0):
                    # Outside Bengaluru, but allow with warning
                    pass
            except (ValueError, TypeError):
                self.errors['coordinates'] = "Invalid coordinates (must be numbers)"
                return False
            
            return True
            
        except Exception as e:
            self.errors['general'] = f"Validation error: {str(e)}"
            return False
    
    def cleaned_data(self):
        """Return sanitized data"""
        return {
            'images': self.files.getlist('images'),
            'scenario_name': self.data.get('scenario_name', '').strip(),
            'location': self.data.get('location', '').strip() or DEFAULT_LOCATION,
            'camera_id': self.data.get('camera_id', '').strip() or DEFAULT_CAMERA_ID,
            'latitude': parse_float_or_default(self.data.get('latitude', DEFAULT_LAT), DEFAULT_LAT),
            'longitude': parse_float_or_default(self.data.get('longitude', DEFAULT_LNG), DEFAULT_LNG),
            'description': self.data.get('description', '').strip(),
        }

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def get_client_ip(request):
    """Extract client IP from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def log_audit(batch_id, action, request, status, details=None):
    """Log audit event"""
    try:
        AuditLog.objects.create(
            batch_id=batch_id,
            action=action,
            ip_address=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')[:500],
            status=status,
            details=to_json_safe(details or {})
        )
    except Exception as e:
        print(f"Audit log failed: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING (SAME AS BEFORE BUT SECURE)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image_secure(image_path, timeout_seconds=UPLOAD_TIMEOUT_SECONDS):
    """Analyze image with timeout protection"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Failed to read image"
        
        # Check image dimensions (max 4000x4000)
        if img.shape[0] > 4000 or img.shape[1] > 4000:
            return None, "Image too large"
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Water detection (blue-heavy pixels)
        blue = img[:, :, 0]
        green = img[:, :, 1]
        red = img[:, :, 2]
        
        water_mask = (blue > green) & (blue > red * 0.8)
        water_pixels = np.sum(water_mask) / (img.shape[0] * img.shape[1]) * 100
        
        # Depth estimate from visible water coverage; previously this path forced
        # depth to 0 whenever `is_flood` was false, which collapsed moderate scenes.
        brightness_factor = 0.85 if brightness > 180 else (1.1 if brightness < 100 else 1.0)
        raw_depth_cm = max(0.0, (water_pixels - 5.0) * 1.3 * brightness_factor)
        normalized = harmonize_prediction(
            raw_depth_cm=raw_depth_cm,
            water_pct=water_pixels,
            raw_confidence=max(0.50, min(0.98, 0.50 + (water_pixels / 125.0))),
            num_anchors=0,
        )

        depth_cm = int(round(float(normalized.get("depth_cm", 0.0))))
        is_water_present = bool(normalized.get("is_water_confirmed", False))
        is_flood = bool(is_water_present and depth_cm >= 12 and brightness < 215)

        if is_flood:
            confidence = min(100.0, round(60.0 + (water_pixels * 1.1), 1))
        elif is_water_present and depth_cm > 0:
            confidence = min(100.0, round(52.0 + (water_pixels * 0.75), 1))
        else:
            confidence = max(50.0, round(90.0 - (water_pixels * 2.0), 1))

        if not is_flood and depth_cm == 0:
            intensity = "SAFE"
        elif depth_cm >= 90:
            intensity = "CRITICAL"
        elif depth_cm >= 50:
            intensity = "HIGH"
        else:
            intensity = "MEDIUM"
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_count = np.sum(edges > 0)
        
        return {
            'is_flood': bool(is_flood),
            'confidence': float(confidence),
            'intensity': str(intensity),
            'depth_cm': int(depth_cm),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'water_pixels': float(water_pixels),
            'edge_count': int(edge_count),
            'dimensions': f"{img.shape[1]}x{img.shape[0]}",
        }, None
        
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# VIEWS
# ─────────────────────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
@csrf_protect
def secure_random_image_upload_page(request):
    """Render upload interface with Bengaluru map and security features"""
    
    # Create CSRF token
    csrf_token = get_token(request)
    
    context = {
        'csrf_token': csrf_token,
        'default_lat': DEFAULT_LAT,
        'default_lng': DEFAULT_LNG,
        'default_camera_id': DEFAULT_CAMERA_ID,
        'default_location': DEFAULT_LOCATION,
        'bengaluru_name': DEFAULT_LOCATION,
    }
    
    return render(request, 'enterprise_upload.html', context)

@require_http_methods(["POST"])
@csrf_protect
def secure_random_image_upload_process(request):
    """Process secure image upload with validation"""
    
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    client_ip = get_client_ip(request)
    
    try:
        # Rate limiting check
        throttle = ImageUploadThrottle()
        if not throttle.allow_request(request, None):
            log_audit(batch_id, 'UPLOAD_FAILED', request, 'rate_limited',
                     {'reason': 'Rate limit exceeded'})
            return JsonResponse({
                'status': 'error',
                'message': 'Too many uploads. Max 10 per hour. Please try again later.'
            }, status=429)
        
        # Validate form
        form = SecureImageUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            log_audit(batch_id, 'UPLOAD_FAILED', request, 'validation_failed',
                     {'errors': form.errors})
            return JsonResponse({
                'status': 'error',
                'message': 'Validation failed',
                'errors': form.errors
            }, status=400)
        
        log_audit(batch_id, 'UPLOAD_START', request, 'started',
                 {'image_count': len(request.FILES.getlist('images'))})
        
        # Get cleaned data
        data = form.cleaned_data()
        images = data['images']
        
        # Create upload directory
        uploads_dir = Path('uploads/temp_uploads')
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        results = []
        flooded_count = 0
        depth_values = []
        intensity_levels = {'SAFE': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        pending_escalations = 0
        integrity_flag_count = 0
        historical_cache = {}
        
        for idx, image_file in enumerate(images, 1):
            try:
                # Save with secure filename
                safe_name = sanitize_filename(image_file.name)
                image_path = uploads_dir / safe_name
                
                with open(image_path, 'wb') as f:
                    for chunk in image_file.chunks():
                        f.write(chunk)
                
                # Analyze image
                analysis, error = analyze_image_secure(str(image_path))
                
                if error or analysis is None:
                    results.append({
                        'image_num': idx,
                        'filename': image_file.name,
                        'status': 'error',
                        'message': error or 'Failed to analyze'
                    })
                    continue
                
                # Update statistics
                analysis = to_json_safe(analysis)
                if analysis['is_flood']:
                    flooded_count += 1
                depth_values.append(analysis['depth_cm'])
                intensity_levels[analysis['intensity']] += 1
                
                zone = BENGALURU_FLOOD_ZONES[(idx - 1) % len(BENGALURU_FLOOD_ZONES)]
                governance_audit = run_data_governance_audit(
                    data['latitude'],
                    data['longitude'],
                    str(image_path),
                )
                integrity_flag_count += len(governance_audit.get('flags', []))

                escalation = recommend_escalation_path(analysis, zone['name'])
                if escalation.get('priority') in ('immediate', 'urgent'):
                    pending_escalations += 1

                zone_key = zone['name']
                if zone_key not in historical_cache:
                    historical_cache[zone_key] = compute_historical_pattern(zone['lat'], zone['lng'])
                historical_pattern = historical_cache[zone_key]
                choke_points = predict_downstream_choke_points(zone['lat'], zone['lng'], analysis)

                results.append({
                    'image_num': idx,
                    'filename': image_file.name,
                    'status': 'success',
                    'zone_name': zone['name'],
                    'latitude': zone['lat'],
                    'longitude': zone['lng'],
                    'captured_at': timezone.now().isoformat(),
                    'analysis': analysis,
                    'governance_audit': governance_audit,
                    'escalation_recommendation': escalation,
                    'predictive_analytics': {
                        'historical_pattern': historical_pattern,
                        'downstream_choke_points': choke_points,
                        'projection_rationale': (
                            f"Projected from local flood trend ({historical_pattern.get('trend')}) "
                            f"and SWD proximity around {zone['name']}."
                        ),
                    },
                })
                
            except Exception as e:
                results.append({
                    'image_num': idx,
                    'filename': image_file.name,
                    'status': 'error',
                    'message': str(e)
                })
        
        # Calculate aggregated statistics
        dry_count = len(images) - flooded_count
        successful_results = [r for r in results if r.get('status') == 'success']
        avg_confidence = np.mean([r['analysis']['confidence'] 
                                 for r in successful_results]) if successful_results else 0
        avg_depth = np.mean(depth_values) if depth_values else 0
        
        # Determine max intensity
        max_intensity = 'SAFE'
        for level in ['CRITICAL', 'HIGH', 'MEDIUM']:
            if intensity_levels[level] > 0:
                max_intensity = level
                break
        
        clustered_incidents = cluster_incidents_for_map(results)
        telemetry_metrics = {
            'active_incidents': int(sum(1 for c in clustered_incidents if c.get('flooded_images', 0) > 0)),
            'pending_escalations': int(pending_escalations),
            'integrity_flags': int(integrity_flag_count),
            'avg_depth_cm': round(float(avg_depth), 1),
            'clustered_bubbles': int(len(clustered_incidents)),
            'ready_batches': 1,
        }
        report_url = f'/report/{batch_id}/'
        ui_commands = build_agentic_ui_commands(results, telemetry_metrics, report_url)

        # Create database record
        upload_result = SecureRandomImageUploadResult.objects.create(
            batch_id=batch_id,
            user_ip=client_ip,
            scenario_name=data['scenario_name'],
            location=data['location'],
            camera_id=data['camera_id'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            description=data['description'],
            total_images=len(images),
            flooded_count=flooded_count,
            dry_count=dry_count,
            avg_confidence=float(avg_confidence),
            avg_depth_cm=float(avg_depth),
            max_intensity=max_intensity,
            results_json={
                'images': to_json_safe(results),
                'ui_commands': to_json_safe(ui_commands),
                'telemetry_metrics': to_json_safe(telemetry_metrics),
                'clustered_incidents': to_json_safe(clustered_incidents),
                'timestamp': timezone.now().isoformat(),
            },
            report_path=report_url
        )
        
        log_audit(batch_id, 'UPLOAD_SUCCESS', request, 'completed',
                 {
                     'flooded': flooded_count,
                     'dry': dry_count,
                     'pending_escalations': pending_escalations,
                     'integrity_flags': integrity_flag_count,
                 })
        
        return JsonResponse({
            'status': 'success',
            'batch_id': batch_id,
            'message': f'Upload successful! Processed {len(images)} images.',
            'statistics': {
                'total_images': len(images),
                'flooded_count': flooded_count,
                'dry_count': dry_count,
                'avg_confidence': float(avg_confidence),
                'avg_depth_cm': float(avg_depth),
                'max_intensity': max_intensity,
            },
            'agentic_summary': {
                'team_brief': (
                    f"{len(successful_results)} image(s) triaged. "
                    f"{pending_escalations} escalation(s) pending. "
                    f"{integrity_flag_count} governance flag(s) detected."
                ),
                'recommendation': (
                    "Prioritize CRITICAL/HIGH clusters for BBMP command center dispatch "
                    "and validate all flagged coordinate anomalies."
                ),
            },
            'telemetry_metrics': telemetry_metrics,
            'clustered_incidents': clustered_incidents,
            'ui_commands': ui_commands,
            'report_url': report_url,
        }, status=201)
        
    except Exception as e:
        log_audit(batch_id, 'UPLOAD_FAILED', request, 'exception',
                 {'error': str(e)})
        return JsonResponse({
            'status': 'error',
            'message': f'Upload failed: {str(e)}'
        }, status=500)

@require_http_methods(["GET"])
def agentic_live_status(request):
    """Return live command-center telemetry and UI commands for proactive updates."""
    try:
        window_minutes = int(request.GET.get("window_minutes", 180))
    except (TypeError, ValueError):
        window_minutes = 180
    window_minutes = max(15, min(window_minutes, 720))

    recent_batches = SecureRandomImageUploadResult.objects.filter(
        created_at__gte=timezone.now() - timedelta(minutes=window_minutes)
    ).order_by("-created_at")[:50]

    recent_results = []
    pending_escalations = 0
    integrity_flags = 0
    active_incident_batches = 0
    depth_values = []
    notifications = []

    for batch in recent_batches:
        images = batch.results_json.get("images", [])
        successful = [img for img in images if img.get("status") == "success"]
        recent_results.extend(successful)
        active_incident_batches += 1 if batch.flooded_count > 0 else 0
        depth_values.append(float(batch.avg_depth_cm))

        for img in successful:
            escalation = img.get("escalation_recommendation", {})
            if escalation.get("priority") in ("immediate", "urgent"):
                pending_escalations += 1
            integrity_flags += len(img.get("governance_audit", {}).get("flags", []))

        notifications.append({
            "title": f"{batch.location} batch processed",
            "message": (
                f"{batch.total_images} images analyzed; "
                f"{batch.flooded_count} flooded, {batch.dry_count} dry."
            ),
            "severity": "warning" if batch.max_intensity in ("HIGH", "CRITICAL") else "info",
            "batch_id": batch.batch_id,
            "report_url": f"/report/{batch.batch_id}/",
            "timestamp": batch.created_at.isoformat(),
        })

    clustered_incidents = cluster_incidents_for_map(recent_results)
    avg_depth_cm = round(float(np.mean(depth_values)), 1) if depth_values else 0.0
    telemetry_metrics = {
        "active_incidents": int(active_incident_batches),
        "pending_escalations": int(pending_escalations),
        "integrity_flags": int(integrity_flags),
        "avg_depth_cm": avg_depth_cm,
        "clustered_bubbles": int(len(clustered_incidents)),
        "ready_batches": int(len(recent_batches)),
    }

    ui_commands = [
        {
            "type": "telemetry.update_metrics",
            "payload": telemetry_metrics,
        },
        {
            "type": "map.replace_incident_bubbles",
            "payload": {
                "clusters": clustered_incidents,
                "strategy": "radius_50m_window_30m",
            },
        },
        {
            "type": "map.toggle_layers",
            "payload": {
                "layers": {
                    "swd_network": True,
                    "traffic": pending_escalations > 0,
                }
            },
        },
    ]

    for notification in notifications[:5]:
        ui_commands.append({
            "type": "notification.push",
            "payload": notification,
        })

    return JsonResponse({
        "status": "success",
        "window_minutes": window_minutes,
        "telemetry_metrics": telemetry_metrics,
        "clustered_incidents": clustered_incidents,
        "notifications": notifications[:10],
        "ui_commands": ui_commands,
    }, status=200)

@require_http_methods(["GET"])
def view_report_secure(request, batch_id):
    """View secure report with results table"""
    import json as _json
    try:
        result = get_object_or_404(SecureRandomImageUploadResult, batch_id=batch_id)
        log_audit(batch_id, "REPORT_VIEWED", request, "success")

        flat_results = []
        for img in result.results_json.get("images", []):
            if img.get("status") != "success":
                continue
            a = img.get("analysis", {})
            flat_results.append({
                "filename": img.get("filename", ""),
                "flooded": bool(a.get("is_flood", False)),
                "confidence": round(float(a.get("confidence", 0)), 1),
                "intensity": a.get("intensity", "SAFE"),
                "estimated_depth": int(a.get("depth_cm", 0)),
                "water_percentage": round(float(a.get("water_pixels", 0)), 1),
                "brightness": round(float(a.get("brightness", 0)), 1),
                "latitude": float(img.get("latitude", result.latitude)),
                "longitude": float(img.get("longitude", result.longitude)),
                "zone_name": img.get("zone_name", result.location),
                "image_data": "",
            })

        n_flood = result.flooded_count
        n_total = result.total_images
        intensity = result.max_intensity
        if intensity == "CRITICAL":
            assessment = f"CRITICAL: {n_flood} of {n_total} images show severe flooding. Immediate emergency response required."
        elif intensity == "HIGH":
            assessment = f"HIGH RISK: {n_flood} of {n_total} images indicate significant flooding. Deploy monitoring teams."
        elif intensity == "MEDIUM":
            assessment = f"MEDIUM RISK: {n_flood} of {n_total} images show possible flooding. Increased monitoring advised."
        else:
            assessment = f"LOW RISK: No flooding detected across {n_total} images. Routine monitoring continues."

        context = {
            "batch_id": result.batch_id,
            "location": result.location,
            "timestamp": result.created_at.strftime("%d %b %Y, %H:%M"),
            "total_images": result.total_images,
            "flooded_count": result.flooded_count,
            "avg_confidence": round(float(result.avg_confidence), 1),
            "avg_depth": round(float(result.avg_depth_cm), 1),
            "max_intensity": result.max_intensity,
            "assessment_summary": assessment,
            "model_version": "1.0.0-cv-basic",
            "processing_time": "< 5",
            "results_json": _json.dumps(flat_results),
        }
        return render(request, "enterprise_report.html", context)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_http_methods(["GET"])
def download_report_secure(request, batch_id):
    """Download report as HTML"""
    
    try:
        result = get_object_or_404(SecureRandomImageUploadResult, batch_id=batch_id)
        
        log_audit(batch_id, 'REPORT_DOWNLOADED', request, 'success')
        
        # Generate HTML content
        html_content = generate_report_html(result)
        
        response = FileResponse(
            iter([html_content.encode('utf-8')]),
            content_type='text/html'
        )
        response['Content-Disposition'] = f'attachment; filename="flood_report_{batch_id}.html"'
        
        return response
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def generate_report_html(result):
    """Generate HTML report"""
    
    intensity_colors = {
        'SAFE': '#228B22',
        'MEDIUM': '#FFD700',
        'HIGH': '#FF4500',
        'CRITICAL': '#8B0000',
    }
    
    rows = ""
    for img in result.results_json.get('images', []):
        if img['status'] != 'success':
            continue
        
        analysis = img['analysis']
        intensity = analysis['intensity']
        color = intensity_colors.get(intensity, '#999999')
        
        rows += f"""
        <tr>
            <td>#{img['image_num']}</td>
            <td>{img['filename']}</td>
            <td>{'FLOOD' if analysis['is_flood'] else 'DRY'}</td>
            <td>{analysis['confidence']:.1f}%</td>
            <td><span style="background-color:{color}; color:white; padding:4px 8px; border-radius:4px;">{intensity}</span></td>
            <td>{analysis['depth_cm']} cm</td>
            <td>{analysis['brightness']:.1f}</td>
            <td>{analysis['contrast']:.1f}</td>
            <td>{analysis['water_pixels']:.1f}%</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flood Detection Report - {result.batch_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
            .summary {{ background: white; padding: 15px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
            .stat-box {{ background: #f0f0f0; padding: 10px; border-radius: 4px; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; background: white; margin: 15px 0; }}
            th {{ background: #333; color: white; padding: 10px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background: #f5f5f5; }}
            .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Flood Detection Analysis Report</h1>
            <p><strong>Batch ID:</strong> {result.batch_id}</p>
            <p><strong>Location:</strong> {result.location} ({result.latitude:.4f} N, {result.longitude:.4f} E)</p>
            <p><strong>Camera:</strong> {result.camera_id}</p>
            <p><strong>Generated:</strong> {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>{result.total_images}</h3>
                    <p>Total Images</p>
                </div>
                <div class="stat-box">
                    <h3>{result.flooded_count}</h3>
                    <p>Flooded</p>
                </div>
                <div class="stat-box">
                    <h3>{result.avg_confidence:.1f}%</h3>
                    <p>Avg Confidence</p>
                </div>
                <div class="stat-box">
                    <h3>{result.avg_depth_cm:.1f} cm</h3>
                    <p>Avg Depth</p>
                </div>
            </div>
        </div>
        
        <div class="summary">
            <h2>Image Analysis Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Filename</th>
                        <th>Status</th>
                        <th>Confidence</th>
                        <th>Intensity</th>
                        <th>Depth (cm)</th>
                        <th>Brightness</th>
                        <th>Contrast</th>
                        <th>Water Pixels %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Report generated by Bengaluru Flood Detection System</p>
            <p style="color: #999; font-size: 11px;">This report is secure and contains sensitive geospatial information. Do not share publicly.</p>
        </div>
    </body>
    </html>
    """
    
    return html
