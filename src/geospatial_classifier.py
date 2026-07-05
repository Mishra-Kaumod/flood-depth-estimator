"""
src/geospatial_classifier.py
============================
Deliverable 2: Geospatial Flood Intensity Classifier.

Translates a raw flood depth (cm) into a 1-5 severity integer and maps
that to a rich colour/metadata structure usable by:

  * Folium  (marker_color, hex_color)
  * GeoJSON / AWS Location Service  (feature properties with RGBA)
  * Leaflet custom legend  (css_color)

All thresholds read from config/config.yaml  aggregator.thresholds.
A standalone Bengaluru GeoJSON FeatureCollection builder is included.

Usage
-----
    from src.geospatial_classifier import FloodIntensityClassifier

    clf = FloodIntensityClassifier()
    band = clf.classify(depth_cm=75.0)
    print(band.severity)          # 4
    print(band.hex_color)         # "#ef4444"
    print(band.next_action)       # "ALERT_TRAFFIC_MANAGEMENT"

    fc = clf.to_geojson(predictions)   # list[dict] → GeoJSON dict
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

# ── Config ─────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


_CFG = _load_cfg()
_THRESHOLDS = _get(_CFG, "aggregator", "thresholds") or {}

# Depth thresholds in cm (read from config, Bengaluru defaults)
_BAND_CM = [
    float(_THRESHOLDS.get("advisory_cm",  10.0)),   # severity 1 → 2
    float(_THRESHOLDS.get("warning_cm",   30.0)),   # severity 2 → 3
    float(_THRESHOLDS.get("alert_cm",     60.0)),   # severity 3 → 4
    float(_THRESHOLDS.get("critical_cm", 100.0)),   # severity 4 → 5
]


# ── Colour palette (5-colour intensity scale) ──────────────────────────────

@dataclass(frozen=True)
class IntensityBand:
    """Immutable descriptor for one severity band."""
    severity: int                   # 1 (lowest) … 5 (highest)
    label: str                      # Human label
    hex_color: str                  # CSS / Leaflet / Folium hex
    rgba: tuple                     # (R, G, B, A) for AWS Location Service
    folium_color: str               # Named colour for folium.Icon
    next_action: str                # Municipal action code
    description: str                # Long-form description for dashboards


_INTENSITY_BANDS: List[IntensityBand] = [
    IntensityBand(
        severity=1,
        label="NORMAL",
        hex_color="#22c55e",
        rgba=(34, 197, 94, 200),
        folium_color="green",
        next_action="MONITOR",
        description="No significant flooding. Standard surveillance.",
    ),
    IntensityBand(
        severity=2,
        label="ADVISORY",
        hex_color="#facc15",
        rgba=(250, 204, 21, 200),
        folium_color="beige",
        next_action="ADVISORY_INCREASE_MONITORING",
        description="Minor flooding (10–30 cm). Heighten sensor polling.",
    ),
    IntensityBand(
        severity=3,
        label="WARNING",
        hex_color="#f97316",
        rgba=(249, 115, 22, 200),
        folium_color="orange",
        next_action="WARNING_PUMP_ACTIVATION",
        description="Moderate flooding (30–60 cm). Activate stormwater pumps.",
    ),
    IntensityBand(
        severity=4,
        label="ALERT",
        hex_color="#ef4444",
        rgba=(239, 68, 68, 200),
        folium_color="red",
        next_action="ALERT_TRAFFIC_MANAGEMENT",
        description="Severe flooding (60–100 cm). Close roads, reroute traffic.",
    ),
    IntensityBand(
        severity=5,
        label="CRITICAL",
        hex_color="#7c3aed",
        rgba=(124, 58, 237, 200),
        folium_color="purple",
        next_action="DEPLOY_EMERGENCY_DIVERSION",
        description="Critical flooding (> 100 cm). Evacuate — emergency diversion.",
    ),
]

# Convenience lookup
_BAND_BY_SEVERITY = {b.severity: b for b in _INTENSITY_BANDS}


# ── Classifier ─────────────────────────────────────────────────────────────

class FloodIntensityClassifier:
    """
    Maps raw flood depth (cm) to a 1-5 IntensityBand.

    Parameters are read from config/config.yaml at instantiation time so
    that updating thresholds in YAML and restarting is sufficient to change
    the classification behaviour — no code edits required.

    Output array layout
    -------------------
    classify() returns one IntensityBand with:
        [severity, label, hex_color, rgba, folium_color, next_action, description]

    to_intensity_array() returns a plain list of the above values for
    direct consumption by visualisation canvas code.
    """

    def __init__(self) -> None:
        self._bands = _INTENSITY_BANDS
        self._thresholds = _BAND_CM   # sorted ascending

    def classify(self, depth_cm: float) -> IntensityBand:
        """Return the IntensityBand for the given depth in centimetres."""
        if depth_cm < self._thresholds[0]:
            return _BAND_BY_SEVERITY[1]
        elif depth_cm < self._thresholds[1]:
            return _BAND_BY_SEVERITY[2]
        elif depth_cm < self._thresholds[2]:
            return _BAND_BY_SEVERITY[3]
        elif depth_cm < self._thresholds[3]:
            return _BAND_BY_SEVERITY[4]
        else:
            return _BAND_BY_SEVERITY[5]

    def to_intensity_array(self, depth_cm: float) -> list:
        """
        Return a flat list representation of the band — suitable for
        direct injection into a Bengaluru spatial visualisation canvas.

        Format:
            [severity(int), label(str), hex_color(str),
             rgba(tuple), folium_color(str), next_action(str), description(str)]
        """
        b = self.classify(depth_cm)
        return [b.severity, b.label, b.hex_color, b.rgba,
                b.folium_color, b.next_action, b.description]

    def all_bands(self) -> List[IntensityBand]:
        """Return all 5 bands — useful for rendering a Leaflet legend."""
        return list(self._bands)

    # ── GeoJSON export ─────────────────────────────────────────────────────

    def to_geojson(self, predictions: List[dict]) -> dict:
        """
        Build a GeoJSON FeatureCollection from a list of prediction dicts.

        Each dict must contain:  latitude, longitude, depth_cm
        Optional:  camera_id, confidence_score, timestamp

        Compatible with:
          * AWS Location Service  (geofence / tracker APIs)
          * Folium GeoJSON layer  (choropleth overlay)
          * Leaflet GeoJSON layer (L.geoJSON(...))
        """
        features = []
        for p in predictions:
            depth_cm = float(p.get("depth_cm", p.get("estimated_flood_depth", 0) * 100))
            band = self.classify(depth_cm)
            ts = p.get("timestamp", datetime.now(timezone.utc).isoformat())
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(p.get("longitude", 0.0)),
                        float(p.get("latitude", 0.0)),
                    ],
                },
                "properties": {
                    "camera_id":             p.get("camera_id", "unknown"),
                    "depth_cm":              round(depth_cm, 2),
                    "estimated_depth_m":     round(depth_cm / 100.0, 4),
                    "confidence_score":      p.get("confidence_score", p.get("confidence", 0.0)),
                    "severity":              band.severity,
                    "severity_label":        band.label,
                    "hex_color":             band.hex_color,
                    "rgba":                  list(band.rgba),
                    "folium_color":          band.folium_color,
                    "next_action":           band.next_action,
                    "description":           band.description,
                    "timestamp":             ts,
                },
            })
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generated_at":  datetime.now(timezone.utc).isoformat(),
                "total_points":  len(features),
                "legend":        self.leaflet_legend_html(),
                "aws_region":    _get(_CFG, "aws", "region", default="ap-south-1"),
            },
        }

    def leaflet_legend_html(self) -> str:
        """Return self-contained HTML for a Leaflet map legend control."""
        rows = ""
        for b in self._bands:
            rows += (
                f'<div style="margin:3px 0">' 
                f'<span style="display:inline-block;width:13px;height:13px;' 
                f'background:{b.hex_color};border-radius:50%;margin-right:6px;' 
                f'vertical-align:middle"></span>' 
                f'<b>{b.label}</b> &mdash; {b.description}</div>\n'
            )
        return (
            '<div id="flood-legend" style="position:absolute;bottom:30px;right:10px;' 
            'z-index:1000;background:white;padding:12px 16px;border-radius:8px;' 
            'box-shadow:0 2px 8px rgba(0,0,0,.25);font-family:sans-serif;font-size:12px;">' 
            '<b style="display:block;margin-bottom:6px;font-size:13px">' 
            'Bengaluru Flood Severity</b>\n' + rows + '</div>'
        )


# ── Module-level convenience function ──────────────────────────────────────

_DEFAULT_CLF = FloodIntensityClassifier()


def classify_depth(depth_cm: float) -> IntensityBand:
    """Module-level shortcut: classify a depth without instantiating the class."""
    return _DEFAULT_CLF.classify(depth_cm)


def depth_to_color(depth_cm: float) -> str:
    """Return hex colour string for a given depth — one-liner for quick use."""
    return _DEFAULT_CLF.classify(depth_cm).hex_color
