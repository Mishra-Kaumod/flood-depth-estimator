# ingestor.py
"""
Image Ingestor — 15-minute batch queue
=======================================
Watches  inbox/<camera_id>/  for new images.
Every 15 minutes it seals the current batch, moves images to
  processing/<batch_id>/
and puts a BatchJob onto a thread-safe queue for the pipeline.

Folder convention
-----------------
inbox/
  CAM_001_LOC_001_12.9172_77.6228_SilkBoard/   ← folder name encodes metadata
    frame_20260719_143001.jpg
    frame_20260719_143045.jpg
  CAM_002_LOC_002_13.0351_77.5975_Hebbal/
    frame_20260719_143010.jpg

processing/
  batch_20260719_1430/                          ← sealed batch
    CAM_001_.../ ...

archive/
  batch_20260719_1430/                          ← after pipeline succeeds
"""

import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

log = logging.getLogger("ingestor")

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CameraImage:
    """One image from one camera."""
    image_path:    Path
    camera_id:     str
    location_id:   str
    latitude:      float
    longitude:     float
    location_name: str
    captured_at:   str  # ISO timestamp parsed from filename or mtime


@dataclass
class BatchJob:
    """A sealed 15-minute batch ready for the pipeline."""
    batch_id:   str                    # e.g. "batch_20260719_1430"
    batch_dir:  Path                   # processing/<batch_id>/
    images:     List[CameraImage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# Folder-name parser  (CAM_001_LOC_001_12.9172_77.6228_SilkBoard)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_camera_folder(folder: Path) -> dict:
    """
    Parse metadata from the camera folder name.
    Expected format: <camera_id>_<location_id>_<lat>_<lon>_<name>
    Falls back gracefully if format is different.
    """
    parts = folder.name.split("_")
    try:
        return {
            "camera_id":     parts[0] if len(parts) > 0 else folder.name,
            "location_id":   parts[1] if len(parts) > 1 else "LOC_UNK",
            "latitude":      float(parts[2]) if len(parts) > 2 else 12.9716,
            "longitude":     float(parts[3]) if len(parts) > 3 else 77.5946,
            "location_name": "_".join(parts[4:]) if len(parts) > 4 else folder.name,
        }
    except (ValueError, IndexError):
        log.warning("Could not fully parse folder name: %s", folder.name)
        return {
            "camera_id":     folder.name,
            "location_id":   "LOC_UNK",
            "latitude":      12.9716,
            "longitude":     77.5946,
            "location_name": folder.name,
        }


def _image_timestamp(path: Path) -> str:
    """Try to parse timestamp from filename, else use mtime."""
    stem = path.stem  # e.g. frame_20260719_143001
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%dT%H-%M-%S"):
        parts = stem.split("_", 1)
        if len(parts) == 2:
            try:
                return datetime.strptime(parts[1], fmt).isoformat()
            except ValueError:
                pass
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Ingestor
# ─────────────────────────────────────────────────────────────────────────────
class ImageIngestor:
    """
    Runs in a background thread.
    Every `interval_seconds` (default 900 = 15 min) it:
      1. Scans inbox/ for all images across all camera folders
      2. Creates a BatchJob
      3. Moves images to processing/<batch_id>/
      4. Puts the BatchJob on `job_queue`
    """

    def __init__(
        self,
        inbox_dir:    Path,
        processing_dir: Path,
        archive_dir:  Path,
        job_queue:    queue.Queue,
        interval_seconds: int = 900,   # 15 minutes
    ):
        self.inbox_dir       = Path(inbox_dir)
        self.processing_dir  = Path(processing_dir)
        self.archive_dir     = Path(archive_dir)
        self.job_queue       = job_queue
        self.interval        = interval_seconds
        self._stop_event     = threading.Event()

        for d in (self.inbox_dir, self.processing_dir, self.archive_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._run, name="ingestor", daemon=True)
        t.start()
        log.info("Ingestor started — interval=%ds inbox=%s", self.interval, self.inbox_dir)
        return t

    def stop(self):
        self._stop_event.set()

    # ── Internal ──────────────────────────────────────────────────────────────
    def _run(self):
        while not self._stop_event.is_set():
            try:
                batch = self._seal_batch()
                if batch and batch.images:
                    log.info("Batch %s sealed — %d images", batch.batch_id, len(batch.images))
                    self.job_queue.put(batch)
                else:
                    log.debug("No images in inbox — skipping batch")
            except Exception:
                log.exception("Ingestor error")
            self._stop_event.wait(timeout=self.interval)

    def _seal_batch(self) -> BatchJob | None:
        batch_id  = f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}"
        batch_dir = self.processing_dir / batch_id
        images: List[CameraImage] = []

        # Scan all camera sub-folders in inbox
        cam_folders = [d for d in self.inbox_dir.iterdir() if d.is_dir()]
        if not cam_folders:
            # Also accept images directly in inbox (flat layout)
            cam_folders = [self.inbox_dir]

        for cam_folder in cam_folders:
            meta = _parse_camera_folder(cam_folder)
            dest_cam_dir = batch_dir / cam_folder.name

            for img_path in sorted(cam_folder.glob("*")):
                if img_path.suffix.lower() not in SUPPORTED:
                    continue
                dest_cam_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_cam_dir / img_path.name
                shutil.move(str(img_path), dest_path)

                images.append(CameraImage(
                    image_path    = dest_path,
                    camera_id     = meta["camera_id"],
                    location_id   = meta["location_id"],
                    latitude      = meta["latitude"],
                    longitude     = meta["longitude"],
                    location_name = meta["location_name"],
                    captured_at   = _image_timestamp(dest_path),
                ))

        if not images:
            return None

        return BatchJob(batch_id=batch_id, batch_dir=batch_dir, images=images)

    def archive_batch(self, batch: BatchJob):
        """Call after pipeline succeeds to move batch to archive."""
        dest = self.archive_dir / batch.batch_id
        if batch.batch_dir.exists():
            shutil.move(str(batch.batch_dir), dest)
            log.info("Archived batch %s → %s", batch.batch_id, dest)
