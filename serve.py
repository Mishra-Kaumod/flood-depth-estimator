"""
LitServe adapter that uses the shared event contract and unified pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError

from src.dlq import get_dead_letter_router
from src.event_contract import FloodEvent, FloodFailureEvent
from src.middleware.retry import RetryPolicy
from src.pipeline import execute_event
from src.settings import load_settings_dict

try:
    from litserve import LitServer
except ImportError:
    LitServer = None
    logging.warning("LitServe not installed. Install: pip install litserve")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class InferenceBatch(BaseModel):
    images: List[Dict[str, Any]] = Field(default_factory=list)


def load_config(config_path: str = "config/config.yaml") -> dict:
    return load_settings_dict(config_path=config_path)


class FloodDepthPredictor(LitServer if LitServer else object):
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.retry_policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=6.0)
        self.dlq = get_dead_letter_router()
        logger.info("Unified serve adapter ready")

    def decode_request(self, request: Any) -> List[FloodEvent]:
        payload = request.json() if hasattr(request, "json") else request
        try:
            batch = InferenceBatch.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid request payload: {exc}") from exc

        events: List[FloodEvent] = []
        for row in batch.images:
            event_payload = dict(row)
            # Backward compatibility for older serve payload shape
            if "image_b64" not in event_payload and "data" in event_payload:
                event_payload["image_b64"] = event_payload.pop("data")
            event_payload["source"] = "serve"
            events.append(FloodEvent.model_validate(event_payload))
        return events

    def predict(self, events: List[FloodEvent]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for event in events:
            try:
                result = execute_event(event, retry_policy=self.retry_policy)
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
        avg_depth = (
            round(sum(item["estimated_depth_meters"] for item in successful) / len(successful), 4)
            if successful
            else None
        )
        return {
            "status": "success",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results": results,
            "summary": {
                "total_images": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "avg_estimated_depth_meters": avg_depth,
            },
        }

    def predict_batch(self, batch: List[FloodEvent]) -> Dict[str, Any]:
        return self.encode_response(self.predict(batch))


def main() -> None:
    config = load_config("config/config.yaml")
    litserve_cfg = config.get("inference", {}).get("litserve", {})
    predictor = FloodDepthPredictor("config/config.yaml")

    if not LitServer:
        logger.error("LitServe not available. Install: pip install litserve")
        return

    server = LitServer(
        predictor,
        port=litserve_cfg.get("port", 8000),
        host=litserve_cfg.get("host", "0.0.0.0"),
        max_batch_size=litserve_cfg.get("max_batch_size", 8),
        batch_timeout=litserve_cfg.get("batch_timeout", 0.05),
        workers=litserve_cfg.get("workers", 4),
    )
    server.run()


def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "flood-depth-estimator",
        "version": "2.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    main()
