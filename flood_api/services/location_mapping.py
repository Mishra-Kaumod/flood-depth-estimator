import hashlib
import json
import os


DEFAULT_BENGALURU_CAMERA_POINTS = [
    {"name": "Majestic", "latitude": 12.9762, "longitude": 77.5713},
    {"name": "Koramangala", "latitude": 12.9352, "longitude": 77.6245},
    {"name": "Indiranagar", "latitude": 12.9784, "longitude": 77.6408},
    {"name": "Whitefield", "latitude": 12.9698, "longitude": 77.7500},
    {"name": "Hebbal", "latitude": 13.0358, "longitude": 77.5970},
    {"name": "Yelahanka", "latitude": 13.1007, "longitude": 77.5963},
    {"name": "Jayanagar", "latitude": 12.9250, "longitude": 77.5938},
    {"name": "Electronic City", "latitude": 12.8456, "longitude": 77.6603},
    {"name": "Banashankari", "latitude": 12.9255, "longitude": 77.5468},
    {"name": "Marathahalli", "latitude": 12.9591, "longitude": 77.6974},
]


def _coerce_point(raw_point):
    return {
        "name": str(raw_point["name"]),
        "latitude": float(raw_point["latitude"]),
        "longitude": float(raw_point["longitude"]),
    }


def get_bengaluru_points():
    raw_points = os.getenv("BENGALURU_CAMERA_POINTS_JSON")
    if not raw_points:
        return DEFAULT_BENGALURU_CAMERA_POINTS

    try:
        parsed_points = json.loads(raw_points)
        if not isinstance(parsed_points, list) or not parsed_points:
            return DEFAULT_BENGALURU_CAMERA_POINTS

        normalized = []
        for point in parsed_points:
            if not isinstance(point, dict):
                return DEFAULT_BENGALURU_CAMERA_POINTS
            if "name" not in point or "latitude" not in point or "longitude" not in point:
                return DEFAULT_BENGALURU_CAMERA_POINTS
            normalized.append(_coerce_point(point))
        return normalized
    except (ValueError, TypeError, KeyError):
        return DEFAULT_BENGALURU_CAMERA_POINTS


def safe_float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def pick_bengaluru_point(seed_value):
    points = get_bengaluru_points()
    digest = hashlib.sha256(str(seed_value).encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(points)
    return points[index]


def resolve_upload_location(location_name, latitude, longitude, fallback_seed):
    fallback_point = pick_bengaluru_point(fallback_seed)
    resolved_latitude = latitude if latitude is not None else fallback_point["latitude"]
    resolved_longitude = longitude if longitude is not None else fallback_point["longitude"]
    resolved_location_name = location_name if location_name else f"{fallback_point['name']} (Randomized)"
    return resolved_location_name, resolved_latitude, resolved_longitude

