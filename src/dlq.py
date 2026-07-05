"""
Dead-letter routing for permanently failed events.

Backends:
1. redis list (preferred in cloud)
2. JSONL file fallback (local persistence)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.event_contract import FloodFailureEvent
from src.settings import load_settings_dict

logger = logging.getLogger(__name__)

_DLQ_ROUTER = None


class DeadLetterRouter:
    def __init__(self, config_path: str = "config/config.yaml"):
        cfg = load_settings_dict(config_path=config_path)
        dlq_cfg = cfg.get("event_processing", {}).get("dlq", {})

        self.enabled = bool(dlq_cfg.get("enabled", True))
        self.backend = str(dlq_cfg.get("backend", "redis")).lower()
        self.redis_url = str(dlq_cfg.get("redis_url", cfg.get("aggregator", {}).get("redis_url", "redis://localhost:6379/1")))
        self.redis_list_key = str(dlq_cfg.get("redis_list_key", "flood:dlq:events"))
        self.file_path = Path(dlq_cfg.get("file_path", "logs/dlq_events.jsonl"))
        self._redis = None

        if self.backend == "redis":
            self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis

            self._redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("DLQ router using redis backend at %s", self.redis_url)
        except Exception as exc:
            logger.warning("DLQ redis unavailable (%s). Falling back to file backend.", exc)
            self.backend = "file"
            self._redis = None

    def publish(self, failure: FloodFailureEvent) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "published": False}

        if self.backend == "redis" and self._redis is not None:
            try:
                self._redis.rpush(self.redis_list_key, json.dumps(failure.model_dump(mode="json")))
                return {
                    "enabled": True,
                    "published": True,
                    "backend": "redis",
                    "redis_list_key": self.redis_list_key,
                }
            except Exception as exc:
                logger.error("DLQ redis publish failed; falling back to file: %s", exc)

        return self._publish_file(failure)

    def _publish_file(self, failure: FloodFailureEvent) -> Dict[str, Any]:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(failure.model_dump(mode="json"), default=str) + "\n")
        return {
            "enabled": True,
            "published": True,
            "backend": "file",
            "file_path": str(self.file_path),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }


def get_dead_letter_router() -> DeadLetterRouter:
    global _DLQ_ROUTER
    if _DLQ_ROUTER is None:
        _DLQ_ROUTER = DeadLetterRouter()
    return _DLQ_ROUTER
