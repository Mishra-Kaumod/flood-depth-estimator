"""
API service for camera-triggered flood events.

This carries over the old high-speed camera upload flow while routing all
inference through the cleaned event pipeline.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import Any

from src.aggregator import SensorPayload, SlidingWindowAggregator
from src.event_contract import FloodEvent
from src.llm_judge import LLMJudge
from src.pipeline import execute_event
from src.settings import load_settings_dict
from src.storage import FloodRepository, TelemetryRecord, records_to_dicts
from src.temporal_analysis import TemporalFloodAnalyzer


class FloodApiService:
    def __init__(self, repository: FloodRepository | None = None):
        self.repository = repository or FloodRepository()
        self.aggregator = SlidingWindowAggregator()
        self.temporal_analyzer = TemporalFloodAnalyzer(self.repository)
        self.config = load_settings_dict()
        self.llm_judge_error = None
        self.llm_judge = self._build_llm_judge()

    def _build_llm_judge(self) -> LLMJudge | None:
        try:
            judge_cfg = self.config.get("inference", {}).get("llm_judge", {})
            return LLMJudge(judge_cfg)
        except Exception as exc:
            # Do not fail the entire service if judge config is invalid.
            self.llm_judge_error = str(exc)
            return None

    def process_camera_upload(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        camera_id: str,
        latitude: float,
        longitude: float,
        location_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        self.repository.upsert_camera(
            camera_id=camera_id,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            description=str(metadata.get("description", "")),
        )

        event = FloodEvent(
            source="api",
            camera_id=camera_id,
            latitude=latitude,
            longitude=longitude,
            image_b64=base64.b64encode(image_bytes).decode("ascii"),
            metadata=metadata,
        )
        result = execute_event(event)
        result_payload = result.to_api_response()
        depth_cm = round(result.estimated_depth_meters * 100.0, 2)
        confidence_pct = round(result.confidence_score * 100.0, 2)
        reference_objects = self._reference_objects_from_metadata(result.metadata)
        result_payload["detected_reference_objects"] = reference_objects
        result_payload["visual_cues"] = result.metadata.get("visual_cues", [])

        llm_judge_result = None
        if self.llm_judge is not None:
            result_payload["llm_judge_enabled"] = bool(self.llm_judge.enabled)
            if self.llm_judge.enabled:
                structured = result_payload.get("structured_features", {}) or {}
                prediction = {
                    "depth_cm": depth_cm,
                    "severity_label": result.severity_label,
                    "action_trigger": result.action_trigger,
                    "confidence_pct": confidence_pct,
                    "water_coverage_pct": structured.get("water_coverage_pct"),
                    "reference_depth_cm": structured.get("reference_depth_cm"),
                    "reference_count": structured.get("reference_count"),
                    "largest_water_region_pct": structured.get("largest_water_region_pct"),
                    "largest_water_region_aspect": structured.get("largest_water_region_aspect"),
                    "waterline_pct": structured.get("waterline_pct"),
                    "region_depth_cm": structured.get("region_depth_cm"),
                    "dense_depth_mean": structured.get("dense_depth_mean"),
                    "dense_depth_p90": structured.get("dense_depth_p90"),
                    "dense_depth_p95": structured.get("dense_depth_p95"),
                    "visual_cues": result_payload.get("visual_cues", []),
                    "label_guide": result_payload.get("label_guide"),
                }
                try:
                    llm_judge_result = self.llm_judge.judge(
                        prediction,
                        image_bytes=image_bytes,
                        filename=filename,
                    )
                    result_payload["llm_judge"] = llm_judge_result
                    if llm_judge_result.get("raw_response"):
                        result_payload["llm_judge_raw_response"] = llm_judge_result["raw_response"]
                except Exception as exc:
                    result_payload["llm_judge_error"] = str(exc)

                if llm_judge_result and llm_judge_result.get("prediction_correct") is False and self.llm_judge.apply_corrections:
                    corrected_depth = float(
                        llm_judge_result.get(
                            "final_depth_cm",
                            llm_judge_result.get("recommended_depth_cm", depth_cm),
                        )
                    )
                    depth_cm = corrected_depth
                    result_payload["corrected_depth_cm"] = corrected_depth
                    result_payload["corrected_severity_label"] = llm_judge_result.get(
                        "final_severity",
                        llm_judge_result.get("recommended_severity", result.severity_label),
                    )
                    result_payload["llm_judge_applied"] = True

            else:
                result_payload["llm_judge_error"] = "LLM judge disabled in config"

            if llm_judge_result is not None:
                result_payload["llm_judge_result"] = llm_judge_result
                judge_depth = self._first_number(
                    llm_judge_result.get("final_depth_cm"),
                    depth_cm if llm_judge_result.get("prediction_correct") else None,
                    llm_judge_result.get("recommended_depth_cm"),
                    depth_cm,
                )
                judge_severity = self._first_text(
                    llm_judge_result.get("final_severity"),
                    result.severity_label if llm_judge_result.get("prediction_correct") else None,
                    llm_judge_result.get("recommended_severity"),
                    result.severity_label,
                )
                result_payload["llm_judge_final_depth_cm"] = judge_depth
                result_payload["llm_judge_final_severity"] = judge_severity
                result_payload["llm_judge_decision_source"] = (
                    "pipeline" if llm_judge_result.get("prediction_correct") else "judge"
                )
        else:
            result_payload["llm_judge_error"] = self.llm_judge_error or "LLM judge unavailable; no validator instance was created"

        telemetry = self.repository.save_telemetry(
            TelemetryRecord(
                id=None,
                timestamp=result.processed_at.isoformat(),
                image_name=filename,
                camera_id=camera_id,
                strategy_applied=result.method,
                surface_water_confirmed_pct=confidence_pct,
                computed_depth_cm=depth_cm,
                system_confidence_score_pct=confidence_pct,
                detected_reference_objects=reference_objects,
                num_reference_objects=len(reference_objects),
                is_water_confirmed=depth_cm > 0 and confidence_pct >= 40.0,
                safety_risk_assessment=f"{result.severity_label} - {result.action_trigger}",
                metadata={
                    "event": event.model_dump(mode="json"),
                    "result": result_payload,
                    "input_metadata": metadata,
                },
            )
        )

        buffer_state = self._buffer_ingress(event, image_bytes, depth_cm, result.confidence_score)
        temporal_result = None
        if buffer_state.get("burst_ready"):
            temporal_result = self.temporal_analyzer.create_temporal_sequence(camera_id)

        return {
            "status": "processing" if buffer_state.get("burst_ready") else "buffered",
            "telemetry_id": telemetry.id,
            "camera_id": camera_id,
            "queue": buffer_state,
            "result": result_payload,
            "temporal": temporal_result,
        }

    def latest_temporal_sequence(self, camera_id: str) -> dict[str, Any] | None:
        sequence = self.temporal_analyzer.latest_sequence(camera_id)
        if sequence is None:
            return None
        return sequence.__dict__

    def trigger_temporal_analysis(self, camera_id: str, time_window_minutes: int = 15):
        return self.temporal_analyzer.create_temporal_sequence(camera_id, time_window_minutes)

    def camera_stats(self, camera_id: str, hours: int = 24) -> dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        camera = self.repository.get_camera(camera_id)
        stats = self.repository.camera_stats(camera_id, since.isoformat())
        stats.update(
            {
                "camera_name": camera.location_name if camera else f"Location {camera_id}",
                "hours_analyzed": hours,
            }
        )
        return stats

    def recent_telemetry(self, limit: int = 20, camera_id: str | None = None):
        return records_to_dicts(self.repository.recent_telemetry(limit=limit, camera_id=camera_id))

    def _buffer_ingress(
        self,
        event: FloodEvent,
        image_bytes: bytes,
        depth_cm: float,
        confidence: float,
    ) -> dict[str, Any]:
        payload = SensorPayload(
            camera_id=event.camera_id,
            latitude=event.latitude,
            longitude=event.longitude,
            image=image_bytes,
        )
        burst = self.aggregator.push(
            payload=payload,
            depth_cm=depth_cm,
            confidence=confidence,
            event_ts=event.timestamp.isoformat(),
        )
        state = self.aggregator.window_state(event.camera_id)
        if burst is not None:
            state["burst"] = burst.model_dump()
        return state

    def _first_number(self, *values: Any) -> float:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _first_text(self, *values: Any) -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return "unknown"
    def _reference_objects_from_metadata(self, metadata: dict[str, Any]) -> list[str]:
        features = metadata.get("structured_features", {})
        objects = features.get("reference_objects") or features.get("objects") or []
        if isinstance(objects, list):
            return [str(obj.get("class", obj)) if isinstance(obj, dict) else str(obj) for obj in objects]
        return []
