# src/queue/job_queue.py
"""
Reliability Layer — Durable Job Queue with Retry + Dead-Letter
================================================================
Replaces Python's in-memory queue.Queue with Redis-backed durability.

Design:
  - Jobs serialised as JSON and pushed to Redis LIST (RPUSH/BLPOP)
  - Each job carries retry_count and max_retries
  - On failure: job re-queued with backoff up to max_retries
  - After max_retries: moved to dead-letter queue + alert fired
  - On app restart: jobs still in Redis — nothing lost

If Redis is unavailable, falls back to in-memory queue with a WARNING.
"""

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger("queue.job_queue")


@dataclass
class QueuedJob:
    batch_id:    str
    batch_dir:   str               # serialisable path string
    image_paths: list[str]         # list of image file paths
    retry_count: int  = 0
    max_retries: int  = 3
    enqueued_at: str  = field(default_factory=lambda: datetime.now().isoformat())
    last_error:  str  = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "QueuedJob":
        return cls(**json.loads(s))


class RedisJobQueue:
    """
    Redis-backed durable queue.
    Falls back to threading.Queue if Redis is unavailable.
    """

    def __init__(
        self,
        redis_url:         str = "redis://localhost:6379/0",
        queue_name:        str = "floodwatch:jobs",
        dead_letter_name:  str = "floodwatch:dead",
        max_retries:       int = 3,
        retry_delay_s:     int = 30,
        on_dead_letter=None,      # callback(QueuedJob) when job hits dead letter
    ):
        self.queue_name       = queue_name
        self.dead_letter_name = dead_letter_name
        self.max_retries      = max_retries
        self.retry_delay_s    = retry_delay_s
        self.on_dead_letter   = on_dead_letter
        self._redis           = None
        self._fallback        = queue.Queue()

        self._connect(redis_url)

    def _connect(self, redis_url: str):
        try:
            import redis as redis_lib
            client = redis_lib.from_url(redis_url, socket_connect_timeout=3,
                                         decode_responses=True)
            client.ping()
            self._redis = client
            log.info("Redis queue connected: %s → %s", redis_url, self.queue_name)
        except Exception:
            log.warning(
                "Redis unavailable — using in-memory queue. "
                "Jobs will be LOST on restart. Fix Redis for production.",
                exc_info=False
            )

    # ── Public API ────────────────────────────────────────────────────────────
    def push(self, job: QueuedJob):
        if self._redis:
            self._redis.rpush(self.queue_name, job.to_json())
        else:
            self._fallback.put(job)

    def pop(self, timeout_s: int = 5) -> Optional[QueuedJob]:
        """Blocking pop. Returns None on timeout."""
        if self._redis:
            result = self._redis.blpop(self.queue_name, timeout=timeout_s)
            if result:
                _, raw = result
                return QueuedJob.from_json(raw)
            return None
        else:
            try:
                return self._fallback.get(timeout=timeout_s)
            except queue.Empty:
                return None

    def requeue_with_backoff(self, job: QueuedJob, error: str):
        """Called by worker when a job fails. Retries or dead-letters."""
        job.retry_count += 1
        job.last_error   = error

        if job.retry_count >= self.max_retries:
            self._dead_letter(job)
        else:
            delay = self.retry_delay_s * (2 ** (job.retry_count - 1))
            log.warning(
                "Job %s failed (attempt %d/%d) — retrying in %ds: %s",
                job.batch_id, job.retry_count, self.max_retries, delay, error
            )
            threading.Thread(
                target=self._delayed_push, args=(job, delay), daemon=True
            ).start()

    def _delayed_push(self, job: QueuedJob, delay_s: int):
        time.sleep(delay_s)
        self.push(job)
        log.info("Job %s re-queued after %ds backoff", job.batch_id, delay_s)

    def _dead_letter(self, job: QueuedJob):
        log.error(
            "Job %s moved to dead-letter after %d failures. Last error: %s",
            job.batch_id, job.retry_count, job.last_error
        )
        if self._redis:
            self._redis.rpush(self.dead_letter_name, job.to_json())
        if self.on_dead_letter:
            try:
                self.on_dead_letter(job)
            except Exception:
                log.exception("Dead-letter callback failed")

    def depth(self) -> int:
        if self._redis:
            return self._redis.llen(self.queue_name)
        return self._fallback.qsize()

    def dead_letter_depth(self) -> int:
        if self._redis:
            return self._redis.llen(self.dead_letter_name)
        return 0

    def peek_dead_letters(self, count: int = 10) -> list[QueuedJob]:
        if self._redis:
            items = self._redis.lrange(self.dead_letter_name, 0, count - 1)
            return [QueuedJob.from_json(i) for i in items]
        return []
