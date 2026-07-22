#!/usr/bin/env python3
"""
scripts/train_severity_model.py
================================
Trains a GradientBoostingRegressor on the labeled CSV produced by
scripts/build_training_set_from_feedback.py.

Features (X):  water_coverage_pct, mean_flood_depth_cm, p90_flood_depth_cm,
               max_flood_depth_cm, calibration_confidence
Label   (y):   corrected_depth_cm  (human-verified ground truth)

Confidence is estimated as the normalised prediction interval width from
two additional quantile regressors (10th and 90th percentile).  Narrower
interval = higher confidence:
    conf = 1 − clip((q90 − q10) / max_depth_in_train, 0, 1)

The three models (mean + q10 + q90) are saved in a single joblib bundle
at the path SeverityStage expects.

Usage:
    python scripts/train_severity_model.py \
        --data   data/severity_training_set.csv \
        --output models/severity_gbr.joblib \
        --test-split 0.2

Install:
    pip install scikit-learn joblib pandas
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("train_severity")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

FEATURE_COLS = [
    "water_coverage_pct",
    "mean_flood_depth_cm",
    "p90_flood_depth_cm",
    "max_flood_depth_cm",
    "calibration_confidence",
]
LABEL_COL = "corrected_depth_cm"


def load_data(csv_path: Path):
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pip install pandas") from e
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    return X, y


def train(X, y, test_split: float):
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
    except ImportError as e:
        raise ImportError("pip install scikit-learn") from e

    if test_split > 0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_split, random_state=42
        )
    else:
        X_tr, X_te, y_tr, y_te = X, X, y, y

    # ── Mean regressor (primary depth estimate) ────────────────────────────
    log.info("Training mean GBR on %d samples…", len(X_tr))
    gbr_mean = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    gbr_mean.fit(X_tr, y_tr)

    # ── Quantile regressors for prediction intervals ───────────────────────
    log.info("Training Q10 and Q90 quantile regressors…")
    gbr_q10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.10,
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    gbr_q90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.90,
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    gbr_q10.fit(X_tr, y_tr)
    gbr_q90.fit(X_tr, y_tr)

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred = gbr_mean.predict(X_te)
    mae    = mean_absolute_error(y_te, y_pred)
    rmse   = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    # Prediction interval coverage on test set
    q10_pred = gbr_q10.predict(X_te)
    q90_pred = gbr_q90.predict(X_te)
    coverage = float(np.mean((y_te >= q10_pred) & (y_te <= q90_pred)))

    log.info("── Evaluation ──────────────────────")
    log.info("  MAE:              %.2f cm", mae)
    log.info("  RMSE:             %.2f cm", rmse)
    log.info("  80%% PI coverage: %.1f%%  (target ≥ 80%%)", coverage * 100)

    metrics = {
        "mae_cm": round(mae, 2),
        "rmse_cm": round(rmse, 2),
        "pi_coverage_80": round(coverage, 3),
        "n_train": len(X_tr),
        "n_test":  len(X_te),
        "feature_cols": FEATURE_COLS,
    }

    # max depth in training set — used to normalise PI width → confidence
    max_depth_in_train = float(np.max(y_tr)) if len(y_tr) > 0 else 200.0

    bundle = {
        "gbr_mean":          gbr_mean,
        "gbr_q10":           gbr_q10,
        "gbr_q90":           gbr_q90,
        "feature_cols":      FEATURE_COLS,
        "max_depth_in_train": max_depth_in_train,
        "metrics":           metrics,
    }
    return bundle, metrics


def save(bundle: dict, out_path: Path) -> None:
    try:
        import joblib
    except ImportError as e:
        raise ImportError("pip install joblib") from e
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    # Save metrics alongside model for traceability
    metrics_path = out_path.with_suffix(".metrics.json")
    with metrics_path.open("w") as f:
        json.dump(bundle["metrics"], f, indent=2)
    log.info("Model saved to %s", out_path)
    log.info("Metrics saved to %s", metrics_path)


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",        default="data/severity_training_set.csv")
    parser.add_argument("--output",      default="models/severity_gbr.joblib")
    parser.add_argument("--test-split",  type=float, default=0.2)
    opts = parser.parse_args(args)

    data_path = Path(opts.data)
    if not data_path.exists():
        log.error("Training data not found: %s", data_path)
        log.error("Run scripts/build_training_set_from_feedback.py first.")
        return 1

    X, y = load_data(data_path)
    log.info("Loaded %d samples from %s", len(y), data_path)

    bundle, metrics = train(X, y, opts.test_split)
    save(bundle, Path(opts.output))

    log.info("Done. MAE=%.2f cm, RMSE=%.2f cm", metrics["mae_cm"], metrics["rmse_cm"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
