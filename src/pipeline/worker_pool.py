# src/pipeline/worker_pool.py
"""
Scalable Pipeline Worker Pool
================================
Replaces single-threaded PipelineWorker with a configurable thread pool.
Multiple workers drain the job queue in parallel.

Each worker:
  1. Pops a QueuedJob from RedisJobQueue
  2. Runs PipelineRunner.run_batch()
  3. Writes all predictions to PostgreSQL (async — separate thread)
  4. Fires alerts via AlertEngine
  5. Instruments Prometheus metrics
  6. On failure → requeue_with_backoff()
  7. Archives batch folder on success

Scale workers via PIPELINE_WORKERS=N in .env
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("pipeline.worker_pool")


class PipelineWorkerPool:
    """
    Manages N worker threads, all reading from the same RedisJobQueue.
    Thread-safe — each worker has its own pipeline runner instance.
    """

    def __init__(
        self,
        n_workers:      int,
        job_queue,                  # RedisJobQueue
        runner_factory: Callable,   # () → PipelineRunner  (called once per worker)
        db_writer,                  # PostgresWriter
        alert_engine,               # AlertEngine
        ingestor,                   # ImageIngestor (for archive_batch)
        metrics,                    # FloodWatchMetrics
    ):
        self.n_workers     = n_workers
        self.job_queue     = job_queue
        self.runner_factory= runner_factory
        self.db_writer     = db_writer
        self.alert_engine  = alert_engine
        self.ingestor      = ingestor
        self.metrics       = metrics
        self._stop_event   = threading.Event()
        self._threads: list[threading.Thread] = []

        # Async DB write queue — pipeline never waits for DB
        self._db_queue: list = []
        self._db_lock  = threading.Lock()
        self._db_thread: Optional[threading.Thread] = None

    # ── Public ────────────────────────────────────────────────────────────────
    def start(self):
        # Start async DB writer thread
        self._db_thread = threading.Thread(
            target=self._db_writer_loop, name="db-writer", daemon=True
        )
        self._db_thread.start()

        # Start N pipeline worker threads
        for i in range(self.n_workers):
            runner = self.runner_factory()
            t = threading.Thread(
                target=self._worker_loop,
                args=(runner, i),
                name=f"pipeline-worker-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        log.info("Worker pool started — %d workers + 1 async DB writer", self.n_workers)

    def stop(self):
        log.info("Stopping worker pool…")
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=30)
        if self._db_thread:
            self._db_thread.join(timeout=10)

    # ── Worker loop ───────────────────────────────────────────────────────────
    def _worker_loop(self, runner, worker_id: int):
        log.info("Worker %d ready", worker_id)
        while not self._stop_event.is_set():
            job = self.job_queue.pop(timeout_s=5)
            if job is None:
                continue

            start = time.time()
            try:
                log.info("[W%d] Processing batch %s (%d images)",
                         worker_id, job.batch_id, len(job.image_paths))

                # Rebuild BatchJob from QueuedJob
                from ingestor import BatchJob, CameraImage
                batch_dir = Path(job.batch_dir)
                batch = _rebuild_batch(job, batch_dir)

                # Run pipeline
                predictions = runner.run_batch(batch)

                # Queue DB writes asynchronously
                with self._db_lock:
                    self._db_queue.extend(predictions)

                # Fire alerts for any high-risk result
                for pred in predictions:
                    self.alert_engine.check_and_fire(pred)

                # Metrics
                elapsed = time.time() - start
                self.metrics.images_processed.inc(len(predictions))
                self.metrics.batch_size.observe(len(predictions))
                self.metrics.queue_depth.set(self.job_queue.depth())
                for pred in predictions:
                    self.metrics.risk_counter.labels(
                        risk_level=pred.risk_level
                    ).inc()

                log.info("[W%d] Batch %s done in %.1fs — %d predictions",
                         worker_id, job.batch_id, elapsed, len(predictions))

                # Archive
                self.ingestor.archive_batch(batch)

            except Exception as e:
                elapsed = time.time() - start
                self.metrics.images_failed.inc()
                log.exception("[W%d] Batch %s failed after %.1fs",
                              worker_id, job.batch_id, elapsed)
                self.job_queue.requeue_with_backoff(job, str(e))

    # ── Async DB writer ───────────────────────────────────────────────────────
    def _db_writer_loop(self):
        """Drains the in-process DB queue — never blocks pipeline workers."""
        log.info("Async DB writer ready")
        while not self._stop_event.is_set():
            batch = []
            with self._db_lock:
                if self._db_queue:
                    batch = self._db_queue[:]
                    self._db_queue.clear()

            if batch:
                try:
                    written = self.db_writer.upsert_batch(batch)
                    self.metrics.db_writes.inc(written)
                    log.debug("DB writer: %d/%d written", written, len(batch))
                except Exception:
                    self.metrics.db_write_errors.inc(len(batch))
                    log.exception("Async DB writer error")
            else:
                time.sleep(0.5)


def _rebuild_batch(job, batch_dir: Path):
    """Reconstruct a BatchJob from a QueuedJob after Redis round-trip."""
    from ingestor import BatchJob, CameraImage
    from datetime import datetime

    images = []
    for img_path_str in job.image_paths:
        img_path = Path(img_path_str)
        # Parse camera metadata from parent folder name
        cam_folder = img_path.parent
        parts = cam_folder.name.split("_")
        try:
            cam_meta = {
                "camera_id":     parts[0] if len(parts) > 0 else cam_folder.name,
                "location_id":   parts[1] if len(parts) > 1 else "LOC_UNK",
                "latitude":      float(parts[2]) if len(parts) > 2 else 12.9716,
                "longitude":     float(parts[3]) if len(parts) > 3 else 77.5946,
                "location_name": "_".join(parts[4:]) if len(parts) > 4 else cam_folder.name,
            }
        except (ValueError, IndexError):
            cam_meta = {"camera_id": cam_folder.name, "location_id": "LOC_UNK",
                        "latitude": 12.9716, "longitude": 77.5946, "location_name": cam_folder.name}

        images.append(CameraImage(
            image_path    = img_path,
            camera_id     = cam_meta["camera_id"],
            location_id   = cam_meta["location_id"],
            latitude      = cam_meta["latitude"],
            longitude     = cam_meta["longitude"],
            location_name = cam_meta["location_name"],
            captured_at   = datetime.now().isoformat(),
        ))

    return BatchJob(batch_id=job.batch_id, batch_dir=batch_dir, images=images)
