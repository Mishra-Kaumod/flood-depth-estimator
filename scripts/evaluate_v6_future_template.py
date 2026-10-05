#!/usr/bin/env python3
"""Future V6 evaluation-report template.

It consumes precomputed prediction rows only. It never invokes inference or trains
models, and it rejects rows marked as CHALLENGE unless explicitly allowed later.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REQUIRED = ("split", "image_id", "actual_depth_cm", "v5_final_depth_cm", "v6_primary_depth_cm", "v6_refined_depth_cm")


def bucket(value: float) -> str:
    if value <= 10: return "0-10"
    if value <= 20: return "10-20"
    if value <= 50: return "20-50"
    if value <= 75: return "50-75"
    return "75+"


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    error, absolute = prediction - actual, np.abs(prediction - actual)
    result = {
        "mae_cm": float(absolute.mean()), "rmse_cm": float(np.sqrt(np.mean(error ** 2))),
        "severe_under_count": int((error < -20).sum()), "severe_over_count": int((error > 20).sum()),
        "actual_gt20_pred_lt5": int(((actual > 20) & (prediction < 5)).sum()),
        "actual_gt30_pred_lt10": int(((actual > 30) & (prediction < 10)).sum()), "bucket_mae_cm": {},
    }
    for name in ("0-10", "10-20", "20-50", "50-75", "75+"):
        mask = np.array([bucket(value) == name for value in actual])
        result["bucket_mae_cm"][name] = None if not mask.any() else float(absolute[mask].mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Precomputed V5/V6 evaluation template")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("development", "final_evaluation"), default="development")
    parser.add_argument("--candidate-freeze", help="Required existing frozen-candidate artifact for final Challenge evaluation")
    args = parser.parse_args()
    with Path(args.predictions).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(name not in rows[0] for name in REQUIRED):
        raise ValueError(f"prediction CSV requires: {', '.join(REQUIRED)}")
    contains_challenge = any(row["split"] == "CHALLENGE" for row in rows)
    if contains_challenge:
        candidate_freeze = Path(args.candidate_freeze) if args.candidate_freeze else None
        # The prediction CSV can contain multiple splits, so apply the same
        # default-deny policy directly rather than infer a manifest filename.
        if args.mode != "final_evaluation" or candidate_freeze is None or not candidate_freeze.is_file():
            raise RuntimeError("CHALLENGE rows are forbidden unless mode=final_evaluation and --candidate-freeze exists")
    actual = np.array([float(row["actual_depth_cm"]) for row in rows])
    result = {"rows": len(rows), "systems": {}}
    for field in ("v5_final_depth_cm", "v6_primary_depth_cm", "v6_refined_depth_cm"):
        present = [row for row in rows if row[field].strip()]
        if present:
            result["systems"][field] = metrics(np.array([float(row["actual_depth_cm"]) for row in present]), np.array([float(row[field]) for row in present]))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
