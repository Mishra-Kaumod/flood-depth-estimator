from __future__ import annotations
import base64, io, logging
from celery import Celery
from PIL import Image

logger = logging.getLogger(__name__)
REDIS_URL = "redis://localhost:6379/0"

celery_app = Celery("flood_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json", result_serializer="json",
    accept_content=["json"],
    task_routes={"tasks.infer_flood_depth": {"queue": "flood_inference"}},
    task_soft_time_limit=25, task_time_limit=30, worker_prefetch_multiplier=1,
)


def _has_significant_water(image, threshold=0.05):
    try:
        import cv2, numpy as np
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([80, 50, 50], dtype=np.uint8)
        upper = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return float(mask.mean()) / 255.0 >= threshold
    except Exception:
        return True


@celery_app.task(bind=True, max_retries=3, soft_time_limit=25)
def infer_flood_depth(self, camera_id, image_b64, latitude, longitude):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2)

    if not _has_significant_water(image):
        return {
            "camera_id": camera_id, "latitude": latitude, "longitude": longitude,
            "estimated_depth_meters": 0.0, "model_confidence_score": 0.99,
            "dynamic_next_action_trigger": "NO_FLOOD_DETECTED",
            "method": "binary_gate_rejected",
        }

    try:
        from app import _predict_single
        pred = _predict_single(image)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)

    result = {
        "camera_id": camera_id, "latitude": latitude, "longitude": longitude,
        "estimated_depth_meters": round(pred["depth_cm"] / 100.0, 4),
        "depth_cm": pred["depth_cm"],
        "model_confidence_score": pred.get("model_confidence_score", pred.get("confidence", 0.5)),
        "dynamic_next_action_trigger": pred["severity"]["level"],
        "severity": pred["severity"],
        "method": pred["method"],
    }

    try:
        from temporal_aggregator import push_frame
        window_result = push_frame(camera_id=camera_id, depth_cm=pred["depth_cm"],
                                   confidence=result["model_confidence_score"])
        if window_result:
            result["temporal_window"] = window_result
    except Exception:
        pass

    return result
