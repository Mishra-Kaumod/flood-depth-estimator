"""
Phase 5: GeoJSON export helper + Leaflet severity legend.
Used by app.py /export/geojson endpoint.
"""
from __future__ import annotations

from datetime import datetime
from typing import List

SEVERITY_COLORS = {
    "NO_FLOOD_DETECTED": "#94a3b8",
    "MONITOR":           "#22c55e",
    "ADVISORY":          "#facc15",
    "WARNING":           "#f97316",
    "ALERT":             "#ef4444",
    "CRITICAL_EVACUATE": "#7c3aed",
}

SEVERITY_ORDER = list(SEVERITY_COLORS.keys())


def depth_to_severity(depth_cm: float) -> str:
    if depth_cm < 1:
        return "NO_FLOOD_DETECTED"
    elif depth_cm < 10:
        return "MONITOR"
    elif depth_cm < 30:
        return "ADVISORY"
    elif depth_cm < 60:
        return "WARNING"
    elif depth_cm < 100:
        return "ALERT"
    else:
        return "CRITICAL_EVACUATE"


def build_geojson(predictions: List[dict]) -> dict:
    features = []
    for p in predictions:
        severity = p.get("dynamic_next_action_trigger") or depth_to_severity(p.get("depth_cm", 0))
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [p.get("longitude", 0.0), p.get("latitude", 0.0)],
            },
            "properties": {
                "camera_id": p.get("camera_id", "unknown"),
                "depth_cm": round(p.get("depth_cm", 0.0), 2),
                "estimated_depth_meters": round(p.get("estimated_depth_meters", p.get("depth_cm", 0.0) / 100.0), 4),
                "model_confidence_score": p.get("model_confidence_score", 0.0),
                "severity": severity,
                "color": SEVERITY_COLORS.get(severity, "#94a3b8"),
                "timestamp": p.get("timestamp", datetime.utcnow().isoformat()),
            },
        })
    return {"type": "FeatureCollection", "features": features}


LEAFLET_LEGEND_HTML = """
<div id="flood-legend" style="position:absolute;bottom:30px;right:10px;z-index:1000;
     background:white;padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.3);
     font-family:sans-serif;font-size:13px;min-width:180px;">
  <b style="display:block;margin-bottom:8px;">Flood Severity</b>
  <div><span style="display:inline-block;width:14px;height:14px;background:#94a3b8;border-radius:50%;margin-right:6px;vertical-align:middle;"></span>No flood</div>
  <div><span style="display:inline-block;width:14px;height:14px;background:#22c55e;border-radius:50%;margin-right:6px;vertical-align:middle;"></span>Monitor (&lt;10 cm)</div>
  <div><span style="display:inline-block;width:14px;height:14px;background:#facc15;border-radius:50%;margin-right:6px;vertical-align:middle;"></span>Advisory (10–30 cm)</div>
  <div><span style="display:inline-block;width:14px;height:14px;background:#f97316;border-radius:50%;margin-right:6px;vertical-align:middle;"></span>Warning (30–60 cm)</div>
  <div><span style="display:inline-block;width:14px;height:14px;background:#ef4444;border-radius:50%;margin-right:6px;vertical-align:middle;"></span>Alert (60–100 cm)</div>
  <div><span style="display:inline-block;width:14px;height:14px;background:#7c3aed;border-radius:50%;margin-right:6px;vertical-align:middle;"></span>Critical (&gt;100 cm)</div>
</div>
"""
