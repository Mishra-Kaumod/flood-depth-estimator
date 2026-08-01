"""
Temporal flood depth analyzer migrated from the old Django app.

It validates water presence across a camera sequence using multiple images,
reference anchors, water consensus, and depth consistency.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from src.storage import FloodRepository, TemporalSequenceRecord


class TemporalFloodAnalyzer:
    REFERENCE_HEIGHTS = {
        "person": {"total_height": 175.0, "torso": 60.0, "legs": 90.0},
        "car": {"total_height": 150.0, "wheel_height": 60.0, "hood_height": 80.0},
        "bus": {"total_height": 300.0, "wheel_height": 100.0, "window_height": 220.0},
        "motorcycle": {"total_height": 100.0, "wheel_height": 55.0, "seat_height": 75.0},
        "truck": {"total_height": 250.0, "wheel_height": 90.0, "cabin_height": 200.0},
        "wall": {"assumed_height": 200.0},
    }

    MIN_ANCHORS_FOR_CONFIDENCE = {
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    WATER_PROBABILITY_THRESHOLD = 0.4

    def __init__(self, repository: FloodRepository | None = None):
        self.repository = repository or FloodRepository()
        self.valid_reference_objects = set(self.REFERENCE_HEIGHTS.keys())

    def get_recent_images_for_camera(self, camera_id: str, minutes: int = 15):
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return self.repository.telemetry_since(camera_id, since.isoformat())

    def validate_water_presence(self, records) -> dict[str, Any]:
        if not records:
            return {
                "is_valid": False,
                "reason": "No images in sequence",
                "num_images": 0,
                "water_consensus": False,
            }

        all_detected_objects = []
        water_detections = []
        depths = []

        for record in records:
            all_detected_objects.extend(record.detected_reference_objects or [])
            water_detections.append(record.surface_water_confirmed_pct / 100.0)
            depths.append(record.computed_depth_cm)

        num_images = len(records)
        unique_objects = set(all_detected_objects)
        num_unique_anchors = len(unique_objects)
        water_consensus_pct = float(np.mean(water_detections)) if water_detections else 0.0
        water_consensus = water_consensus_pct >= self.WATER_PROBABILITY_THRESHOLD

        if num_unique_anchors >= self.MIN_ANCHORS_FOR_CONFIDENCE["high"]:
            confidence_level = "high"
            is_valid = water_consensus and num_images >= 2
        elif num_unique_anchors >= self.MIN_ANCHORS_FOR_CONFIDENCE["medium"]:
            confidence_level = "medium"
            is_valid = water_consensus and num_images >= 3
        elif num_unique_anchors >= self.MIN_ANCHORS_FOR_CONFIDENCE["low"]:
            confidence_level = "low"
            is_valid = water_consensus and num_images >= 5
        else:
            confidence_level = "insufficient"
            is_valid = False

        depth_std = float(np.std(np.array(depths))) if len(depths) > 1 else 0.0

        return {
            "is_valid": is_valid,
            "num_images": num_images,
            "unique_anchor_objects": list(unique_objects),
            "num_unique_anchors": num_unique_anchors,
            "water_consensus_pct": round(water_consensus_pct * 100, 2),
            "water_consensus": water_consensus,
            "confidence_level": confidence_level,
            "reason": self._get_validation_reason(
                is_valid,
                num_unique_anchors,
                water_consensus,
                num_images,
                water_consensus_pct,
            ),
            "depth_consistency_std": round(depth_std, 2),
        }

    def calculate_multi_anchor_depth(self, records) -> dict[str, Any]:
        if not records:
            return {"error": "No records provided"}

        depth_estimates: dict[str, list[dict[str, Any]]] = {}

        for record in records:
            for obj_type in record.detected_reference_objects or []:
                if obj_type not in self.REFERENCE_HEIGHTS:
                    continue
                depth_estimates.setdefault(obj_type, []).append(
                    {
                        "depth_cm": record.computed_depth_cm,
                        "confidence": record.system_confidence_score_pct / 100.0,
                        "timestamp": record.timestamp,
                    }
                )

        aggregated = {}
        for obj_type, measurements in depth_estimates.items():
            depths = [m["depth_cm"] for m in measurements]
            confidences = [max(m["confidence"], 0.001) for m in measurements]
            weighted_depth = float(np.average(depths, weights=confidences))
            aggregated[obj_type] = {
                "mean_depth_cm": round(float(np.mean(depths)), 2),
                "weighted_depth_cm": round(weighted_depth, 2),
                "std_dev_cm": round(float(np.std(depths)), 2),
                "min_depth_cm": round(float(np.min(depths)), 2),
                "max_depth_cm": round(float(np.max(depths)), 2),
                "num_measurements": len(measurements),
                "avg_confidence": round(float(np.mean(confidences)), 3),
            }

        return aggregated

    def create_temporal_sequence(self, camera_id: str, time_window_minutes: int = 15):
        records = self.get_recent_images_for_camera(camera_id, time_window_minutes)
        if len(records) < 2:
            return {
                "status": "insufficient_data",
                "message": f"Only {len(records)} image(s) in sequence. Need at least 2.",
                "camera_id": camera_id,
            }

        validation = self.validate_water_presence(records)
        depth_estimates = self.calculate_multi_anchor_depth(records)
        unique_anchors = list(
            set(obj for record in records for obj in (record.detected_reference_objects or []))
        )

        average_depth_cm = None
        max_depth_cm = None
        min_depth_cm = None
        if depth_estimates and "error" not in depth_estimates and validation["is_valid"]:
            all_depths = [v["weighted_depth_cm"] for v in depth_estimates.values()]
            average_depth_cm = round(float(np.mean(all_depths)), 2)
            max_depth_cm = round(float(np.max(all_depths)), 2)
            min_depth_cm = round(float(np.min(all_depths)), 2)

        sequence = self.repository.save_temporal_sequence(
            TemporalSequenceRecord(
                id=None,
                camera_id=camera_id,
                sequence_start=records[0].timestamp,
                sequence_end=records[-1].timestamp,
                image_count=len(records),
                average_depth_cm=average_depth_cm,
                max_depth_cm=max_depth_cm,
                min_depth_cm=min_depth_cm,
                water_detected_in_images=sum(
                    1 for r in records if r.surface_water_confirmed_pct >= 40
                ),
                detected_anchor_types=unique_anchors,
                consensus_water_present=validation["water_consensus"],
                confidence_score=self._calculate_confidence_score(validation),
                telemetry_record_ids=[r.id for r in records if r.id is not None],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )

        return {
            "status": "success" if validation["is_valid"] else "warning",
            "sequence_id": sequence.id,
            "camera_id": camera_id,
            "num_images": len(records),
            "time_span_minutes": self._minutes_between(records[0].timestamp, records[-1].timestamp),
            "validation": validation,
            "depth_estimates_by_anchor": depth_estimates,
            "consensus_depth_cm": sequence.average_depth_cm,
            "final_risk_assessment": self._assess_risk(sequence.average_depth_cm, validation),
        }

    def latest_sequence(self, camera_id: str):
        return self.repository.latest_temporal_sequence(camera_id)

    def _get_validation_reason(
        self,
        is_valid: bool,
        num_anchors: int,
        water_consensus: bool,
        num_images: int,
        water_consensus_pct: float,
    ) -> str:
        if not water_consensus:
            return f"Water not confirmed across images ({round(water_consensus_pct * 100, 2)}% average confidence)"
        if num_anchors == 0:
            return "No reference objects detected - cannot validate depth"
        if num_anchors == 1 and num_images < 5:
            return f"Only 1 reference object type detected. Need 5+ images, got {num_images}"
        if num_anchors == 2 and num_images < 3:
            return f"Only 2 reference object types. Need 3+ images, got {num_images}"
        if is_valid:
            return f"VALIDATED: {num_anchors} anchor types across {num_images} images"
        return "Insufficient data to validate"

    def _calculate_confidence_score(self, validation: dict[str, Any]) -> float:
        factors = [
            min(validation["num_unique_anchors"] / 3.0, 1.0) * 0.4,
            (validation["water_consensus_pct"] / 100.0) * 0.3,
            min(validation["num_images"] / 10.0, 1.0) * 0.3,
        ]
        return round(sum(factors), 3)

    def _assess_risk(self, depth_cm, validation: dict[str, Any]) -> dict[str, str]:
        if depth_cm is None or not validation["is_valid"]:
            return {
                "level": "UNVERIFIED",
                "reason": "Insufficient data to confirm water presence",
            }
        if depth_cm < 15:
            return {"level": "LOW", "reason": f"Shallow depth ({depth_cm}cm)"}
        if depth_cm < 30:
            return {"level": "MODERATE", "reason": f"Depth {depth_cm}cm - small vehicles compromised"}
        if depth_cm < 60:
            return {"level": "HIGH", "reason": f"Depth {depth_cm}cm - most vehicles risk stalling"}
        return {"level": "CRITICAL", "reason": f"Depth {depth_cm}cm - closure recommended"}

    def _minutes_between(self, start_iso: str, end_iso: str) -> float:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        return round((end - start).total_seconds() / 60, 1)
