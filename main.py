# main.py
"""
FloodWatch AI — Main Orchestrator
===================================
Wires together:
  1. ImageIngestor  — watches inbox/, queues 15-min batches
  2. PipelineRunner — runs all 5 model stages per image
  3. PostgresWriter — persists every FloodPrediction
  4. UI             — runs as a separate process (streamlit run ui/app.py)

Usage:
    python main.py                         # uses config/config.yaml
    python main.py --config my_config.yaml
    python main.py --interval 60           # 60-second batches for testing

Architecture (each box is independent / swappable):

  inbox/ folder
       │  (every 15 min)
       ▼
  ┌─────────────┐      queue.Queue      ┌──────────────────────────────┐
  │  Ingestor   │ ──── BatchJob ──────► │       PipelineRunner         │
  └─────────────┘                       │                              │
                                        │  SegFormer → water_mask      │
                                        │  YOLOv8   → ref_objects      │
                                        │  Depth V2 → depth_map        │
                                        │  Fusion   → features         │
                                        │  Severity → FloodPrediction  │
                                        └──────────┬───────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │   PostgresWriter      │
                                        └──────────────────────┘
                                                   │
                                        ┌──────────▼───────────┐
                                        │  Streamlit UI         │  ← separate process
                                        │  (ui/app.py)          │
                                        └──────────────────────┘
"""

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import yaml

from ingestor        import ImageIngestor, BatchJob
from pipeline        import PipelineRunner
from db.postgres     import PostgresWriter, DB_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("main")

# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread — consumes queue, runs pipeline, writes to DB
# ─────────────────────────────────────────────────────────────────────────────
class PipelineWorker(threading.Thread):
    def __init__(self, job_queue: queue.Queue, runner: PipelineRunner,
                 writer: PostgresWriter, ingestor: ImageIngestor):
        super().__init__(name="pipeline-worker", daemon=True)
        self.job_queue = job_queue
        self.runner    = runner
        self.writer    = writer
        self.ingestor  = ingestor
        self._stop     = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        log.info("Pipeline worker started — waiting for batches…")
        while not self._stop.is_set():
            try:
                batch: BatchJob = self.job_queue.get(timeout=5)
            except queue.Empty:
                continue

            try:
                log.info("▶  Processing batch %s (%d images)",
                         batch.batch_id, len(batch.images))

                # Run all 5 pipeline stages for every image in the batch
                predictions = self.runner.run_batch(batch)

                # Write all predictions to PostgreSQL
                written = self.writer.upsert_batch(predictions)
                log.info("✅ Batch %s → %d/%d written to DB",
                         batch.batch_id, written, len(predictions))

                # Move processed images to archive
                self.ingestor.archive_batch(batch)

            except Exception:
                log.exception("Pipeline worker error on batch %s", batch.batch_id)
            finally:
                self.job_queue.task_done()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FloodWatch AI Orchestrator")
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--interval", type=int, default=None,
                        help="Override batch interval in seconds (default: 900)")
    parser.add_argument("--inbox",    default=None,
                        help="Override inbox directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Directories
    base       = Path(cfg.get("base_dir", "."))
    inbox_dir  = Path(args.inbox or cfg.get("inbox_dir",  base / "inbox"))
    proc_dir   = Path(cfg.get("processing_dir", base / "processing"))
    arch_dir   = Path(cfg.get("archive_dir",    base / "archive"))
    interval   = args.interval or cfg.get("batch_interval_seconds", 900)

    log.info("=" * 60)
    log.info("FloodWatch AI  —  Starting")
    log.info("  inbox:    %s", inbox_dir)
    log.info("  interval: %ds  (%.0f min)", interval, interval / 60)
    log.info("  DB:       %s", DB_URL)
    log.info("=" * 60)

    # Shared queue
    job_queue: queue.Queue = queue.Queue()

    # Initialise modules
    ingestor = ImageIngestor(inbox_dir, proc_dir, arch_dir, job_queue, interval)
    runner   = PipelineRunner(cfg)
    writer   = PostgresWriter(DB_URL)
    worker   = PipelineWorker(job_queue, runner, writer, ingestor)

    # Graceful shutdown
    def _shutdown(sig, frame):
        log.info("Shutting down…")
        ingestor.stop()
        worker.stop()
        writer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start all threads
    ingestor.start()
    worker.start()

    log.info("🟢 FloodWatch running.  Drop images into: %s", inbox_dir)
    log.info("   Run UI separately:  streamlit run ui/app.py")

    # Keep main thread alive
    while True:
        time.sleep(10)
        pending = job_queue.qsize()
        if pending:
            log.info("Queue depth: %d batches pending", pending)


if __name__ == "__main__":
    main()
