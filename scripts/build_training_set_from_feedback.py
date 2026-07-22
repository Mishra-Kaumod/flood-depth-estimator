#!/usr/bin/env python3
"""
scripts/build_training_set_from_feedback.py
============================================
Pulls human-corrected depth records from the feedback store and joins them
with the StructuredFeatures that were logged at prediction time.

Outputs a labeled CSV ready for scripts/train_severity_model.py.

Usage:
    python scripts/build_training_set_from_feedback.py \
        --feedback-db   db/feedback.sqlite  \
        --features-log  logs/structured_features.jsonl \
        --output        data/severity_training_set.csv

Feedback store schema (SQLite):
    CREATE TABLE feedback (
        prediction_id  TEXT PRIMARY KEY,
        camera_id      TEXT,
        captured_at    TEXT,
        model_depth_cm REAL,
        corrected_depth_cm REAL NOT NULL,   -- human-verified depth
        corrected_by   TEXT,
        corrected_at   TEXT
    );

StructuredFeatures log format (one JSON per line):
    {"prediction_id": "...", "water_coverage_pct": 12.5,
     "mean_flood_depth_cm": 28.1, "p90_flood_depth_cm": 35.4,
     "max_flood_depth_cm": 42.0, "calibration_source": "yolo_car",
     "calibration_confidence": 0.82}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
from pathlib import Path

log = logging.getLogger("build_training_set")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

FEATURE_COLS = [
    "water_coverage_pct",
    "mean_flood_depth_cm",
    "p90_flood_depth_cm",
    "max_flood_depth_cm",
    "calibration_confidence",
]
LABEL_COL   = "corrected_depth_cm"
OUTPUT_COLS = ["prediction_id", "camera_id", "captured_at"] + FEATURE_COLS + [LABEL_COL]


def load_feedback(db_path: Path) -> dict[str, dict]:
    """Return dict[prediction_id → feedback row]."""
    if not db_path.exists():
        log.warning("Feedback DB not found: %s — returning empty", db_path)
        return {}
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM feedback").fetchall()
    con.close()
    return {r["prediction_id"]: dict(r) for r in rows}


def load_features_log(log_path: Path) -> dict[str, dict]:
    """Return dict[prediction_id → structured features]."""
    if not log_path.exists():
        log.warning("Features log not found: %s — returning empty", log_path)
        return {}
    features: dict[str, dict] = {}
    with log_path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pid = row.get("prediction_id")
                if pid:
                    features[pid] = row
            except json.JSONDecodeError as e:
                log.warning("Line %d: JSON parse error: %s", lineno, e)
    return features


def build_dataset(
    feedback: dict[str, dict],
    features: dict[str, dict],
) -> list[dict]:
    rows = []
    matched = 0
    skipped_no_features = 0
    skipped_missing_cols = 0

    for pid, fb in feedback.items():
        feat = features.get(pid)
        if feat is None:
            skipped_no_features += 1
            continue

        row: dict = {
            "prediction_id": pid,
            "camera_id":     fb.get("camera_id", ""),
            "captured_at":   fb.get("corrected_at", ""),
            LABEL_COL:       fb["corrected_depth_cm"],
        }
        # Copy feature columns
        missing = []
        for col in FEATURE_COLS:
            if col in feat:
                row[col] = feat[col]
            else:
                missing.append(col)
        if missing:
            log.debug("Skipping %s — missing features: %s", pid, missing)
            skipped_missing_cols += 1
            continue

        rows.append(row)
        matched += 1

    log.info(
        "Joined: %d records matched, %d skipped (no features), %d skipped (missing cols)",
        matched, skipped_no_features, skipped_missing_cols,
    )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), out_path)


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-db",   default="db/feedback.sqlite")
    parser.add_argument("--features-log",  default="logs/structured_features.jsonl")
    parser.add_argument("--output",        default="data/severity_training_set.csv")
    opts = parser.parse_args(args)

    feedback = load_feedback(Path(opts.feedback_db))
    features = load_features_log(Path(opts.features_log))

    if not feedback:
        log.error("No feedback records found. Collect corrected predictions first.")
        return 1

    rows = build_dataset(feedback, features)
    if not rows:
        log.error("No matching rows after join. Check that prediction IDs overlap.")
        return 1

    write_csv(rows, Path(opts.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
