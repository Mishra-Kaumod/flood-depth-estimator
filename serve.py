"""
LitServe adapter — wraps PipelineRunner with the shared event contract.

Migrated from src/pipeline.execute_event (archived) to pipeline/runner.py
so all three entry points (api/server.py, ingestor, serve.py) share the
same 6-stage pipeline implementation.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from pipeline.runner     import PipelineRunner
from pipeline.severity   import FloodPrediction
from src.dlq             import get_dead_letter_router
from src.event_contract  import FloodEvent, FloodResultEvent, FloodFailureEvent
from src.middleware.retry import RetryPolicy
from src.settings        import load_settings_dict

try:
    from litserve import LitAPI, LitServer
except ImportError:
    LitAPI = object
    LitServer = None
    logging.warning("LitServe not installed. Install: pip install litserve")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Risk level → FloodResultEvent severity int + color code
_RISK_META = {
    "NO FLOOD":  (1, "#00C853"),
    "LOW RISK":  (2, "#FFD600"),
    "MODERATE":  (3, "#FF6D00"),
    "HIGH RISK": (4, "#D50000"),
    "CRITICAL":  (5, "#7C4DFF"),
}


class InferenceBatch(BaseModel):
    images: List[Dict[str, Any]] = Field(default_factory=list)


def load_config(config_path: str = "config/config.yaml") -> dict:
    return load_settings_dict(config_path=config_path)


def _prediction_to_result(pred: FloodPrediction, event: FloodEvent) -> FloodResultEvent:
    """Convert FloodPrediction → FloodResultEvent (shared response contract)."""
    severity, color = _RISK_META.get(pred.risk_level, (1, "#00C853"))
    return FloodResultEvent(
        event_id               = event.event_id,
        trace_id               = event.trace_id,
        source                 = event.source,
        timestamp              = event.timestamp,
        camera_id              = event.camera_id,
        latitude               = event.latitude,
        longitude              = event.longitude,
        estimated_depth_meters = round(pred.water_depth_cm / 100.0, 4),
        confidence_score       = round(pred.confidence_pct / 100.0, 4),
        color_code             = color,
        action_trigger         = pred.recommended_action,
        severity               = severity,
        severity_label         = pred.risk_level,
        method                 = pred.ensemble_method or "model_only",
        window_frame_count     = 1,
        status                 = "success",
        metadata               = {
            "calibration_source":    pred.calibration_source,
            "seg_engine":            pred.seg_engine,
            "depth_engine":          pred.depth_engine,
            "gemini_agreement_score": pred.gemini_agreement_score,
        },
    )


def _event_to_bgr(event: FloodEvent) -> np.ndarray:
    """Decode base64 image from FloodEvent to BGR numpy array."""
    img_bytes = base64.b64decode(event.image_b64)
    arr       = np.frombuffer(img_bytes, np.uint8)
    img_bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"cv2 could not decode image for camera_id={event.camera_id!r}")
    return img_bgr


class FloodDepthPredictor(LitAPI):

    def setup(self, device):
        config = load_config("config/config.yaml")
        pipeline_cfg = config.get("pipeline", config)   # works with both key layouts
        self.runner  = PipelineRunner({"pipeline": pipeline_cfg})
        self.retry_policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=6.0)
        self.dlq     = get_dead_letter_router()
        logger.info("Unified serve adapter ready (PipelineRunner)")

    def decode_request(self, request: Any) -> List[FloodEvent]:
        payload = request.json() if hasattr(request, "json") else request
        try:
            batch = InferenceBatch.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid request payload: {exc}") from exc

        events: List[FloodEvent] = []
        for row in batch.images:
            event_payload = dict(row)
            # Backward compatibility
            if "image_b64" not in event_payload and "data" in event_payload:
                event_payload["image_b64"] = event_payload.pop("data")
            event_payload["source"] = "serve"
            events.append(FloodEvent.model_validate(event_payload))
        return events

    def predict(self, events: List[FloodEvent]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for event in events:
            try:
                img_bgr = _event_to_bgr(event)
                pred    = self.runner.run_b64_image(
                    img_bgr      = img_bgr,
                    camera_id    = event.camera_id,
                    latitude     = event.latitude,
                    longitude    = event.longitude,
                    location_id  = event.metadata.get("location_id", event.camera_id),
                    location_name= event.metadata.get("location_name", event.camera_id),
                    captured_at  = event.timestamp.isoformat(),
                )
                result = _prediction_to_result(pred, event)
                results.append(result.model_dump(mode="json"))
            except Exception as exc:
                failure = FloodFailureEvent.from_exception(
                    exc=exc,
                    stage="serve.predict",
                    attempts=self.retry_policy.max_attempts,
                    max_attempts=self.retry_policy.max_attempts,
                    retry_exhausted=True,
                    event=event,
                    source="serve",
                    metadata={"adapter": "serve.FloodDepthPredictor.predict"},
                )
                dlq_info = self.dlq.publish(failure)
                failure.metadata["dlq"] = dlq_info
                results.append(failure.to_api_response())
        return results

    def encode_response(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful = [r for r in results if r.get("status") == "success"]
        avg_depth  = (
            round(sum(item["estimated_depth_meters"] for item in successful) / len(successful), 4)
            if successful else None
        )
        return {
            "status":       "success",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results":      results,
            "summary": {
                "total_images":               len(results),
                "successful":                 len(successful),
                "failed":                     len(results) - len(successful),
                "avg_estimated_depth_meters": avg_depth,
            },
        }

    def predict_batch(self, batch: List[FloodEvent]) -> Dict[str, Any]:
        return self.encode_response(self.predict(batch))


def main() -> None:
    config        = load_config("config/config.yaml")
    litserve_cfg  = config.get("inference", {}).get("litserve", {})

    if not LitServer:
        logger.error("LitServe not available. Install: pip install litserve")
        return

    predictor = FloodDepthPredictor(
        max_batch_size = litserve_cfg.get("max_batch_size", 8),
        batch_timeout  = litserve_cfg.get("batch_timeout", 0.05),
    )

    server = LitServer(
        predictor,
        accelerator      = "auto",
        workers_per_device = litserve_cfg.get("workers", 2),
    )
    server.run(
        port = litserve_cfg.get("port", 8000),
        host = litserve_cfg.get("host", "0.0.0.0"),
    )


def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "flood-depth-estimator",
        "version": "2.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    main()
