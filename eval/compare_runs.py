#!/usr/bin/env python3
"""
eval/compare_runs.py
=====================
Diff two benchmark JSON reports and surface regressions / improvements.

Usage:
    python eval/compare_runs.py \
        reports/benchmark_20260720_120000.json \
        reports/benchmark_20260721_120000.json

Outputs a human-readable diff to stdout and exits with code:
  0 = all metrics equal or improved
  1 = regression detected (MAE or RMSE increased beyond threshold)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Default thresholds for declaring a regression
MAE_REGRESSION_CM   = 2.0   # MAE increase > 2cm = regression
RMSE_REGRESSION_CM  = 3.0   # RMSE increase > 3cm = regression
ACC_REGRESSION      = 0.03  # overall accuracy drop > 3pp = regression

RISK_LEVELS = ["NO FLOOD", "LOW RISK", "MODERATE", "HIGH RISK", "CRITICAL"]


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def compare(baseline: dict, current: dict) -> tuple[list[str], bool]:
    """Returns (lines_to_print, regression_detected)."""
    bm = baseline["metrics"]
    cm = current["metrics"]
    lines = []
    regression = False

    def _row(name, bval, cval, worse_if="higher", fmt="{:.2f}"):
        nonlocal regression
        delta = cval - bval
        direction = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        is_worse = (worse_if == "higher" and delta > 0) or (worse_if == "lower" and delta < 0)
        flag = " ⚠️  REGRESSION" if is_worse else ("  ✅ improved" if (
            (worse_if == "higher" and delta < 0) or (worse_if == "lower" and delta > 0)
        ) else "")
        b_str = fmt.format(bval)
        c_str = fmt.format(cval)
        d_str = f"{direction}{abs(delta):{fmt[1:]}}"
        return f"  {name:<30} {b_str:>8} → {c_str:>8}  ({d_str}){flag}"

    lines.append("=" * 70)
    lines.append(f"  BASELINE  {baseline['run_id']}  ({baseline['timestamp'][:10]})")
    lines.append(f"  CURRENT   {current['run_id']}  ({current['timestamp'][:10]})")
    lines.append("=" * 70)
    lines.append("")

    # ── Depth metrics ─────────────────────────────────────────────────────
    lines.append("Depth accuracy:")
    mae_delta  = cm["mae_cm"]  - bm["mae_cm"]
    rmse_delta = cm["rmse_cm"] - bm["rmse_cm"]
    if mae_delta  > MAE_REGRESSION_CM:  regression = True
    if rmse_delta > RMSE_REGRESSION_CM: regression = True
    lines.append(_row("MAE (cm)",  bm["mae_cm"],  cm["mae_cm"]))
    lines.append(_row("RMSE (cm)", bm["rmse_cm"], cm["rmse_cm"]))
    lines.append("")

    # ── Accuracy ──────────────────────────────────────────────────────────
    lines.append("Classification accuracy:")
    acc_delta = cm["overall_accuracy"] - bm["overall_accuracy"]
    if acc_delta < -ACC_REGRESSION: regression = True
    lines.append(_row("Overall accuracy", bm["overall_accuracy"],
                       cm["overall_accuracy"], worse_if="lower", fmt="{:.3f}"))
    lines.append("")

    lines.append("Per-tier accuracy:")
    for tier in RISK_LEVELS:
        b_tier = bm["tier_accuracy"].get(tier, 0)
        c_tier = cm["tier_accuracy"].get(tier, 0)
        lines.append(_row(f"  {tier}", b_tier, c_tier, worse_if="lower", fmt="{:.3f}"))
    lines.append("")

    lines.append("Sample counts:")
    lines.append(f"  Baseline: {bm['n']} samples")
    lines.append(f"  Current:  {cm['n']} samples")
    lines.append("")

    if regression:
        lines.append("🚨  REGRESSION DETECTED — check model or data before deploying.")
    else:
        lines.append("✅  No regressions detected.")
    lines.append("")

    return lines, regression


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline benchmark JSON")
    parser.add_argument("current",  help="Current benchmark JSON")
    parser.add_argument("--mae-threshold",  type=float, default=MAE_REGRESSION_CM)
    parser.add_argument("--rmse-threshold", type=float, default=RMSE_REGRESSION_CM)
    parser.add_argument("--acc-threshold",  type=float, default=ACC_REGRESSION)
    opts = parser.parse_args(args)

    global MAE_REGRESSION_CM, RMSE_REGRESSION_CM, ACC_REGRESSION
    MAE_REGRESSION_CM  = opts.mae_threshold
    RMSE_REGRESSION_CM = opts.rmse_threshold
    ACC_REGRESSION     = opts.acc_threshold

    baseline = load(Path(opts.baseline))
    current  = load(Path(opts.current))
    lines, regression = compare(baseline, current)
    print("\n".join(lines))
    return 1 if regression else 0


if __name__ == "__main__":
    sys.exit(main())
