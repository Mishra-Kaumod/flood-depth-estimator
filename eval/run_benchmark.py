#!/usr/bin/env python3
"""
eval/run_benchmark.py
======================
Loads labeled ground-truth records, runs each through PipelineRunner,
and computes MAE, RMSE, and a 5-class risk-tier confusion matrix.

Outputs a dated JSON + Markdown report in reports/.

Usage:
    python eval/run_benchmark.py \
        --ground-truth  data/severity_training_set.csv \
        --pipeline-cfg  config/config.yaml \
        --output-dir    reports/

Ground-truth CSV format (same as severity training set):
    prediction_id, camera_id, captured_at,
    water_coverage_pct, mean_flood_depth_cm, p90_flood_depth_cm,
    max_flood_depth_cm, calibration_confidence,
    corrected_depth_cm  ← human-verified label

Install:
    pip install scikit-learn pandas pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RISK_LEVELS = ["NO FLOOD", "LOW RISK", "MODERATE", "HIGH RISK", "CRITICAL"]
DEPTH_TO_RISK_TABLE = [
    (0,    "NO FLOOD"),
    (15,   "LOW RISK"),
    (35,   "MODERATE"),
    (60,   "HIGH RISK"),
    (9999, "CRITICAL"),
]


def depth_to_risk(depth_cm: float) -> str:
    for threshold, risk in DEPTH_TO_RISK_TABLE:
        if depth_cm <= threshold:
            return risk
    return "CRITICAL"


def load_ground_truth(csv_path: Path) -> list[dict]:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pip install pandas") from e
    df = pd.read_csv(csv_path)
    required = {"corrected_depth_cm"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")
    return df.to_dict("records")


def run_pipeline_on_features(records: list[dict], cfg: dict) -> list[dict]:
    """
    Run severity stage on pre-computed StructuredFeatures (from CSV).
    This avoids needing image files for benchmark — useful for offline eval.
    """
    from pipeline.severity import SeverityStage
    from pipeline.fusion   import StructuredFeatures
    import numpy as np

    stage    = SeverityStage(cfg.get("pipeline", {}).get("severity_weights"))
    results  = []

    for rec in records:
        # Reconstruct StructuredFeatures from CSV columns
        features = StructuredFeatures(
            water_coverage_pct     = float(rec.get("water_coverage_pct", 0)),
            water_pixel_count      = 0,
            mean_flood_depth_cm    = float(rec.get("mean_flood_depth_cm", 0)),
            max_flood_depth_cm     = float(rec.get("max_flood_depth_cm", 0)),
            p90_flood_depth_cm     = float(rec.get("p90_flood_depth_cm", 0)),
            calibration_source     = rec.get("calibration_source", "fallback"),
            calibration_confidence = float(rec.get("calibration_confidence", 0.3)),
            seg_engine             = "benchmark",
            yolo_engine            = "benchmark",
            depth_engine           = "benchmark",
            water_mask             = np.zeros((1, 1), dtype=bool),
            depth_map_cm           = np.zeros((1, 1), dtype=np.float32),
        )
        pred = stage.predict(
            features      = features,
            location_id   = rec.get("camera_id", "bench"),
            camera_id     = rec.get("camera_id", "bench"),
            latitude      = 0.0,
            longitude     = 0.0,
            location_name = "benchmark",
            timestamp     = rec.get("captured_at", ""),
            batch_id      = "benchmark",
        )
        results.append({
            "prediction_id":    rec.get("prediction_id", ""),
            "camera_id":        rec.get("camera_id", ""),
            "pred_depth_cm":    pred.water_depth_cm,
            "true_depth_cm":    float(rec["corrected_depth_cm"]),
            "pred_risk":        pred.risk_level,
            "true_risk":        depth_to_risk(float(rec["corrected_depth_cm"])),
            "confidence_pct":   pred.confidence_pct,
        })
    return results


def compute_metrics(results: list[dict]) -> dict:
    pred  = [r["pred_depth_cm"]  for r in results]
    true  = [r["true_depth_cm"]  for r in results]
    n     = len(results)

    mae   = sum(abs(p - t) for p, t in zip(pred, true)) / n
    rmse  = math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, true)) / n)

    # 5×5 confusion matrix over risk tiers
    matrix: dict[str, dict[str, int]] = {r: {c: 0 for c in RISK_LEVELS} for r in RISK_LEVELS}
    for r in results:
        matrix[r["true_risk"]][r["pred_risk"]] += 1

    # Per-tier accuracy
    tier_acc: dict[str, float] = {}
    for tier in RISK_LEVELS:
        row_total = sum(matrix[tier].values())
        tier_acc[tier] = (matrix[tier][tier] / row_total) if row_total > 0 else 0.0

    overall_acc = sum(matrix[t][t] for t in RISK_LEVELS) / n

    return {
        "n":           n,
        "mae_cm":      round(mae, 2),
        "rmse_cm":     round(rmse, 2),
        "overall_accuracy": round(overall_acc, 3),
        "tier_accuracy": {k: round(v, 3) for k, v in tier_acc.items()},
        "confusion_matrix": matrix,
    }


def write_report(metrics: dict, cfg_path: str, results: list[dict],
                 output_dir: Path, run_id: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"benchmark_{run_id}.json"
    md_path   = output_dir / f"benchmark_{run_id}.md"

    report = {
        "run_id":    run_id,
        "timestamp": _utcnow(),
        "config":    cfg_path,
        "metrics":   metrics,
    }
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    # Markdown report
    m = metrics
    cm = m["confusion_matrix"]
    md = [
        f"# Benchmark Report — {run_id}",
        f"Generated: {_utcnow()}  |  Config: `{cfg_path}`\n",
        "## Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| N samples | {m['n']} |",
        f"| MAE | **{m['mae_cm']} cm** |",
        f"| RMSE | **{m['rmse_cm']} cm** |",
        f"| Overall Tier Accuracy | {m['overall_accuracy']*100:.1f}% |",
        "",
        "## Per-tier accuracy",
    ] + [f"- {tier}: {acc*100:.1f}%" for tier, acc in m["tier_accuracy"].items()] + [
        "",
        "## Confusion matrix (rows=true, cols=predicted)",
        "| True \\ Pred | " + " | ".join(RISK_LEVELS) + " |",
        "|---" * (len(RISK_LEVELS) + 1) + "|",
    ] + [
        "| " + tier + " | " + " | ".join(str(cm[tier][p]) for p in RISK_LEVELS) + " |"
        for tier in RISK_LEVELS
    ]

    with md_path.open("w") as f:
        f.write("\n".join(md) + "\n")

    return json_path, md_path


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth",  default="data/severity_training_set.csv")
    parser.add_argument("--pipeline-cfg",  default="config/config.yaml")
    parser.add_argument("--output-dir",    default="reports")
    opts = parser.parse_args(args)

    gt_path = Path(opts.ground_truth)
    if not gt_path.exists():
        log.error("Ground truth not found: %s", gt_path)
        return 1

    cfg = _load_yaml(opts.pipeline_cfg)
    log.info("Loading ground truth from %s…", gt_path)
    records = load_ground_truth(gt_path)
    log.info("Loaded %d records", len(records))

    log.info("Running pipeline on structured features…")
    results = run_pipeline_on_features(records, cfg)

    metrics = compute_metrics(results)
    log.info("MAE=%.2f cm  RMSE=%.2f cm  Acc=%.1f%%",
             metrics["mae_cm"], metrics["rmse_cm"], metrics["overall_accuracy"] * 100)

    run_id    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jpath, mpath = write_report(metrics, opts.pipeline_cfg, results,
                                 Path(opts.output_dir), run_id)
    log.info("JSON report: %s", jpath)
    log.info("MD   report: %s", mpath)
    return 0


def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    sys.exit(main())
