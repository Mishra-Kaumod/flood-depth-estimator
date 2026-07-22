#!/usr/bin/env python3
"""
ml_ops/retraining_trigger.py
=============================
Weekly automated benchmark + retraining gate.

Runs eval/run_benchmark.py against a rolling window of recent feedback,
then calls eval/compare_runs.py against the prior week's baseline.
If MAE degrades beyond threshold, automatically triggers
scripts/train_severity_model.py to retrain and saves the new model.

Designed to be run as a GitHub Actions workflow or cron job:
  python ml_ops/retraining_trigger.py --window-days 7

Exit codes:
  0 = no regression, nothing retrained
  1 = regression detected, retraining triggered (check logs)
  2 = retraining succeeded (new model saved, ready to deploy)
  3 = retraining failed (error logged)

GitHub Actions usage (see .github/workflows/model-readiness-gate.yml):
  - run: python ml_ops/retraining_trigger.py --window-days 7 --auto-retrain
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("retraining_trigger")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REPORTS_DIR        = Path("reports")
TRAINING_DATA      = Path("data/severity_training_set.csv")
MODEL_OUTPUT       = Path("models/severity_gbr.joblib")
FEEDBACK_DB        = Path("db/feedback.sqlite")
FEATURES_LOG       = Path("logs/structured_features.jsonl")

# Regression thresholds (override with --mae-threshold etc.)
DEFAULT_MAE_THRESHOLD  = 5.0   # cm
DEFAULT_RMSE_THRESHOLD = 8.0   # cm
DEFAULT_ACC_THRESHOLD  = 0.05  # 5 pp accuracy drop


def find_latest_baseline() -> Path | None:
    """Find the most recent benchmark JSON that is at least 6 days old."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=6)
    reports = sorted(REPORTS_DIR.glob("benchmark_*.json"), reverse=True)
    for p in reports:
        try:
            with p.open() as f:
                d = json.load(f)
            ts = datetime.fromisoformat(d["timestamp"])
            if ts < cutoff:
                return p
        except Exception:
            continue
    return None


def run_benchmark(output_dir: Path) -> Path | None:
    """Run eval/run_benchmark.py and return the new JSON report path."""
    cmd = [
        sys.executable, "eval/run_benchmark.py",
        "--ground-truth", str(TRAINING_DATA),
        "--output-dir",   str(output_dir),
    ]
    log.info("Running benchmark: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    log.info(result.stdout.strip())
    if result.returncode != 0:
        log.error("Benchmark failed:\n%s", result.stderr)
        return None
    # Find the newest benchmark file
    reports = sorted(output_dir.glob("benchmark_*.json"), reverse=True)
    return reports[0] if reports else None


def compare_with_baseline(current: Path, baseline: Path,
                           mae_thr: float, rmse_thr: float, acc_thr: float) -> bool:
    """Returns True if regression detected."""
    cmd = [
        sys.executable, "eval/compare_runs.py",
        str(baseline), str(current),
        "--mae-threshold",  str(mae_thr),
        "--rmse-threshold", str(rmse_thr),
        "--acc-threshold",  str(acc_thr),
    ]
    log.info("Comparing %s vs baseline %s…", current.name, baseline.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    return result.returncode == 1  # 1 = regression


def rebuild_training_set() -> bool:
    cmd = [
        sys.executable, "scripts/build_training_set_from_feedback.py",
        "--feedback-db",  str(FEEDBACK_DB),
        "--features-log", str(FEATURES_LOG),
        "--output",       str(TRAINING_DATA),
    ]
    log.info("Rebuilding training set from feedback…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    log.info(result.stdout.strip())
    if result.returncode != 0:
        log.error("Training set build failed:\n%s", result.stderr)
        return False
    return True


def retrain() -> bool:
    cmd = [
        sys.executable, "scripts/train_severity_model.py",
        "--data",   str(TRAINING_DATA),
        "--output", str(MODEL_OUTPUT),
    ]
    log.info("Retraining severity model…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    log.info(result.stdout.strip())
    if result.returncode != 0:
        log.error("Retraining failed:\n%s", result.stderr)
        return False
    log.info("New model saved to %s", MODEL_OUTPUT)
    return True


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days",    type=int,   default=7)
    parser.add_argument("--auto-retrain",   action="store_true",
                        help="Automatically retrain if regression detected")
    parser.add_argument("--mae-threshold",  type=float, default=DEFAULT_MAE_THRESHOLD)
    parser.add_argument("--rmse-threshold", type=float, default=DEFAULT_RMSE_THRESHOLD)
    parser.add_argument("--acc-threshold",  type=float, default=DEFAULT_ACC_THRESHOLD)
    opts = parser.parse_args(args)

    # ── Step 1: rebuild training set from recent feedback ─────────────────
    if FEEDBACK_DB.exists():
        if not rebuild_training_set():
            log.error("Cannot rebuild training set — aborting")
            return 3

    if not TRAINING_DATA.exists():
        log.warning("No training data available — skipping benchmark")
        return 0

    # ── Step 2: run benchmark ─────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    current_report = run_benchmark(REPORTS_DIR)
    if current_report is None:
        log.error("Benchmark run failed")
        return 3

    # ── Step 3: compare with baseline ─────────────────────────────────────
    baseline = find_latest_baseline()
    if baseline is None:
        log.info("No baseline found — current report becomes the baseline")
        return 0

    regression = compare_with_baseline(
        current_report, baseline,
        opts.mae_threshold, opts.rmse_threshold, opts.acc_threshold,
    )

    if not regression:
        log.info("No regression — no retraining needed")
        return 0

    log.warning("Regression detected!")
    if not opts.auto_retrain:
        log.info("Pass --auto-retrain to trigger automatic retraining")
        return 1

    # ── Step 4: retrain ───────────────────────────────────────────────────
    if not retrain():
        return 3

    log.info("Retraining complete. Deploy %s to activate.", MODEL_OUTPUT)
    return 2


if __name__ == "__main__":
    sys.exit(main())
