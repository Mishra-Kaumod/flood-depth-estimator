"""
Unified flood processing pipeline shared by API, queue worker, and serve adapter.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.aggregator import SensorPayload, SlidingWindowAggregator
from src.event_contract import FloodEvent, FloodResultEvent
from src.geospatial_classifier import FloodIntensityClassifier
from src.middleware.observability import observe_execution
from src.middleware.retry import RetryPolicy, run_with_retry
from src.settings import load_settings_dict

logger = logging.getLogger(__name__)

_PROCESSOR = None


def _has_significant_water(image: Image.Image, threshold: float = 0.05) -> bool:
    try:
        import cv2

        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([80, 50, 50], dtype=np.uint8)
        upper = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return float(mask.mean()) / 255.0 >= threshold
    except Exception:
        return True


class UnifiedEventProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_settings_dict(config_path=config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.transform = self._build_transform()
        self.aggregator = SlidingWindowAggregator()
        self.classifier = FloodIntensityClassifier()

        inference_cfg = self.config.get("inference", {})
        self.use_mc_dropout_confidence = bool(inference_cfg.get("use_mc_dropout_confidence", True))
        self.mc_passes = int(inference_cfg.get("mc_passes", 8))

        model_path = Path(inference_cfg.get("model_path", "models/best_flood_model_water_aware.pth"))
        if model_path.exists():
            self._load_weights(model_path)
        else:
            logger.warning("Model file not found at %s. Using random weights.", model_path)

        self.model.eval()

    def _build_model(self) -> nn.Module:
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        return model

    def _build_transform(self) -> transforms.Compose:
        train_cfg = self.config.get("training", {})
        image_size = tuple(train_cfg.get("image_size", [224, 224]))
        norm_cfg = train_cfg.get(
            "normalization",
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        )
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_cfg["mean"], std=norm_cfg["std"]),
            ]
        )

    def _load_weights(self, model_path: Path) -> None:
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded model weights from %s", model_path)

    def _predict_depth_and_confidence(self, tensor: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            depth_normalized = float(self.model(tensor).squeeze().item())

        confidence = min(max(depth_normalized * 1.1, 0.0), 1.0)
        if self.use_mc_dropout_confidence:
            try:
                from mc_dropout import mc_dropout_confidence

                mean_val, confidence = mc_dropout_confidence(
                    self.model,
                    tensor,
                    n_passes=self.mc_passes,
                )
                depth_normalized = float(mean_val)
            except Exception as exc:
                logger.warning("MC-dropout unavailable. Falling back to heuristic confidence: %s", exc)

        depth_cm = max(0.0, min(depth_normalized * 100.0, 500.0))
        return round(depth_cm, 2), round(float(confidence), 4)

    def process_event(self, event: FloodEvent) -> FloodResultEvent:
        image_bytes = event.image_bytes()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if not _has_significant_water(image):
            band = self.classifier.classify(0.0)
            return FloodResultEvent(
                event_id=event.event_id,
                trace_id=event.trace_id,
                source=event.source,
                timestamp=event.timestamp,
                camera_id=event.camera_id,
                latitude=event.latitude,
                longitude=event.longitude,
                estimated_depth_meters=0.0,
                confidence_score=0.99,
                color_code=band.hex_color,
                action_trigger="NO_FLOOD_DETECTED",
                severity=band.severity,
                severity_label=band.label,
                method="water_gate_rejected",
                window_frame_count=1,
                metadata={"schema_version": event.schema_version},
            )

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        depth_cm, confidence = self._predict_depth_and_confidence(tensor)

        sensor_payload = SensorPayload(
            camera_id=event.camera_id,
            latitude=event.latitude,
            longitude=event.longitude,
            image=image_bytes,
        )
        burst = self.aggregator.push(
            payload=sensor_payload,
            depth_cm=depth_cm,
            confidence=confidence,
            event_ts=event.timestamp.isoformat(),
        )

        if burst is not None:
            depth_m = burst.estimated_flood_depth
            confidence_out = burst.confidence_score
            action = burst.next_action_recommendation
            frame_count = burst.frame_count
            method = "aggregated"
        else:
            depth_m = depth_cm / 100.0
            confidence_out = confidence
            action = self.classifier.classify(depth_cm).next_action
            frame_count = 1
            method = "single_frame"

        band = self.classifier.classify(depth_m * 100.0)
        return FloodResultEvent(
            event_id=event.event_id,
            trace_id=event.trace_id,
            source=event.source,
            timestamp=event.timestamp,
            camera_id=event.camera_id,
            latitude=event.latitude,
            longitude=event.longitude,
            estimated_depth_meters=round(depth_m, 4),
            confidence_score=round(confidence_out, 4),
            color_code=band.hex_color,
            action_trigger=action,
            severity=band.severity,
            severity_label=band.label,
            method=method,
            window_frame_count=frame_count,
            metadata={"schema_version": event.schema_version},
        )


def get_processor() -> UnifiedEventProcessor:
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = UnifiedEventProcessor()
    return _PROCESSOR


def execute_event(
    event: FloodEvent,
    retry_policy: RetryPolicy | None = None,
) -> FloodResultEvent:
    processor = get_processor()
    if retry_policy is None:
        retry_cfg = processor.config.get("event_processing", {}).get("retry", {})
        policy = RetryPolicy(
            max_attempts=int(retry_cfg.get("max_attempts", 3)),
            base_delay_seconds=float(retry_cfg.get("base_delay_seconds", 0.5)),
            max_delay_seconds=float(retry_cfg.get("max_delay_seconds", 8.0)),
            jitter_seconds=float(retry_cfg.get("jitter_seconds", 0.25)),
        )
    else:
        policy = retry_policy

    def operation(attempt: int) -> FloodResultEvent:
        return observe_execution(
            event_id=event.event_id,
            trace_id=event.trace_id,
            camera_id=event.camera_id,
            source=event.source,
            stage="unified_pipeline",
            attempt=attempt,
            operation=lambda: processor.process_event(event),
        )

    return run_with_retry(
        operation=operation,
        policy=policy,
        on_retry=lambda attempt, delay, exc: logger.warning(
            "retrying event=%s trace=%s attempt=%d delay=%.2fs error=%s",
            event.event_id,
            event.trace_id,
            attempt + 1,
            delay,
            exc,
        ),
    )
