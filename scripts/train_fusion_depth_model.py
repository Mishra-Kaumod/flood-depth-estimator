#!/usr/bin/env python3
"""Train a lightweight fusion model from existing pipeline signals.

This model does not replace the image models. It learns how to combine their
outputs into the final flood depth using labeled examples.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.segformer_yolo_depthv2_pipeline import get_segformer_yolo_depthv2_pipeline

FEATURE_NAMES = [
    "pipeline_depth_cm",
    "efficientnet_candidate_depth_cm",
    "reference_depth_cm",
    "reference_count",
    "max_reference_submersion",
    "dense_depth_cm",
    "water_coverage_pct",
    "near_water_coverage_pct",
    "mid_water_coverage_pct",
    "far_water_coverage_pct",
    "largest_water_region_pct",
    "region_depth_cm",
    "waterline_pct",
    "immediate_risk",
    "far_water_only",
    "mask_quality_warning",
    "low_water_gate_applied",
    "shallow_water_gate_exception",
    "muddy_water_fallback_applied",
    "full_road_water_no_reference",
]

BOOL_FEATURES = {
    "immediate_risk",
    "far_water_only",
    "mask_quality_warning",
    "low_water_gate_applied",
    "shallow_water_gate_exception",
    "muddy_water_fallback_applied",
    "full_road_water_no_reference",
}


class FusionDepthModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def read_labels(labels_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            filename = row.get("filename") or row.get("image_name")
            depth = row.get("depth_cm")
            if not filename or depth in (None, ""):
                continue
            rows.append({"filename": filename, "actual_depth_cm": float(depth)})
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def extract_feature_row(image_path: Path, filename: str, actual_depth_cm: float, split: str) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    staged = get_segformer_yolo_depthv2_pipeline().predict(image_rgb)
    structured = staged.get("structured_features", {}) or {}

    pipeline_depth_cm = safe_float(
        structured.get("pre_residual_fusion_depth_cm"),
        safe_float(staged.get("depth_cm")),
    )

    row: dict[str, Any] = {
        "split": split,
        "filename": filename,
        "actual_depth_cm": float(actual_depth_cm),
        "pipeline_depth_cm": pipeline_depth_cm,
        "pipeline_confidence": safe_float(staged.get("confidence")),
    }

    for name in FEATURE_NAMES:
        if name == "pipeline_depth_cm":
            continue
        if name == "dense_depth_cm":
            row[name] = safe_float(structured.get("dense_depth_p90")) * 120.0
        elif name in BOOL_FEATURES:
            row[name] = 1.0 if bool(structured.get(name, False)) else 0.0
        else:
            row[name] = safe_float(structured.get(name))

    return row


def build_feature_cache(args: argparse.Namespace) -> pd.DataFrame:
    split_dir = Path(args.split_dir)
    all_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        labels_csv = split_dir / f"labels_{split}.csv"
        images_dir = split_dir / "images" / split
        labels = read_labels(labels_csv)
        if args.limit_per_split:
            labels = labels[: args.limit_per_split]
        print(f"Extracting {split}: {len(labels)} images")
        for index, label in enumerate(labels, start=1):
            filename = label["filename"]
            image_path = images_dir / filename
            if not image_path.exists():
                print(f"  skip missing: {image_path}")
                continue
            row = extract_feature_row(image_path, filename, label["actual_depth_cm"], split)
            all_rows.append(row)
            if index % args.progress_every == 0 or index == len(labels):
                print(f"  {split}: {index}/{len(labels)}")

    frame = pd.DataFrame(all_rows)
    cache_path = Path(args.features_out)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    print(f"Saved features: {cache_path} ({len(frame)} rows)")
    return frame


def load_or_build_features(args: argparse.Namespace) -> pd.DataFrame:
    cache_path = Path(args.features_out)
    if cache_path.exists() and not args.rebuild_features:
        print(f"Using cached features: {cache_path}")
        return pd.read_csv(cache_path)
    return build_feature_cache(args)


def make_tensors(frame: pd.DataFrame, feature_names: list[str], max_depth_cm: float, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    x = frame[feature_names].astype(float).to_numpy(dtype=np.float32)
    y = np.clip(frame["actual_depth_cm"].astype(float).to_numpy(dtype=np.float32), 0.0, max_depth_cm) / max_depth_cm
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x = (x - mean) / std
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32), mean, std


def predict_cm(model: nn.Module, x: torch.Tensor, max_depth_cm: float, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            batch = x[start : start + 256].to(device)
            pred = model(batch).cpu().numpy().reshape(-1) * max_depth_cm
            preds.append(pred)
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred))) if len(actual) else float("nan")


def train_fusion(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = load_or_build_features(args)
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Feature cache must contain train, val, and test rows.")

    x_train, y_train, mean, std = make_tensors(train_df, FEATURE_NAMES, args.max_depth_cm)
    x_val, y_val, _, _ = make_tensors(val_df, FEATURE_NAMES, args.max_depth_cm, mean, std)
    x_test, y_test, _, _ = make_tensors(test_df, FEATURE_NAMES, args.max_depth_cm, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = FusionDepthModel(len(FEATURE_NAMES)).to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.05)

    best_state: dict[str, Any] | None = None
    best_val_mae = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_pred = predict_cm(model, x_val, args.max_depth_cm, device)
        val_actual = val_df["actual_depth_cm"].astype(float).to_numpy()
        val_mae = mae(val_actual, val_pred)
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss / max(1, len(loader)):.5f} val_mae={val_mae:.2f}cm")

        if val_mae + 0.01 < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch}; best_val_mae={best_val_mae:.2f}cm")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model.")
    model.load_state_dict(best_state)

    for split_name, split_df, x_split in [("train", train_df, x_train), ("val", val_df, x_val), ("test", test_df, x_test)]:
        split_pred = predict_cm(model, x_split, args.max_depth_cm, device)
        actual = split_df["actual_depth_cm"].astype(float).to_numpy()
        print(
            f"{split_name} MAE | "
            f"fusion={mae(actual, split_pred):.2f}cm | "
            f"current_pipeline={mae(actual, split_df['pipeline_depth_cm'].astype(float).to_numpy()):.2f}cm | "
            f"efficientnet={mae(actual, split_df['efficientnet_candidate_depth_cm'].astype(float).to_numpy()):.2f}cm"
        )

    test_pred = predict_cm(model, x_test, args.max_depth_cm, device)
    report = test_df.copy()
    report["fusion_pred_depth_cm"] = np.round(test_pred, 2)
    report["fusion_abs_error_cm"] = np.round(np.abs(report["actual_depth_cm"].astype(float).to_numpy() - test_pred), 2)
    report["pipeline_abs_error_cm"] = np.round(np.abs(report["actual_depth_cm"].astype(float) - report["pipeline_depth_cm"].astype(float)), 2)
    report["efficientnet_abs_error_cm"] = np.round(np.abs(report["actual_depth_cm"].astype(float) - report["efficientnet_candidate_depth_cm"].astype(float)), 2)

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    checkpoint = {
        "model_state_dict": best_state,
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean.astype(float).tolist(),
        "feature_std": std.astype(float).tolist(),
        "max_depth_cm": float(args.max_depth_cm),
        "best_val_mae_cm": float(best_val_mae),
        "training_config": vars(args),
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)

    print(f"Saved fusion model: {model_path}")
    print(f"Saved test report: {report_path}")

    worst = report.sort_values("fusion_abs_error_cm", ascending=False).head(10)
    print("Worst fusion test errors:")
    for _, row in worst.iterrows():
        print(
            f"  {row['filename']}: actual={float(row['actual_depth_cm']):.2f}cm "
            f"fusion={float(row['fusion_pred_depth_cm']):.2f}cm "
            f"pipeline={float(row['pipeline_depth_cm']):.2f}cm "
            f"efficientnet={float(row['efficientnet_candidate_depth_cm']):.2f}cm"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fusion model from pipeline signals")
    parser.add_argument("--split-dir", default="training_runs/20260816_154019")
    parser.add_argument("--features-out", default="reports/fusion_training_features.csv")
    parser.add_argument("--model-out", default="models/candidate/fusion_depth_model.pt")
    parser.add_argument("--report-out", default="reports/fusion_depth_eval.csv")
    parser.add_argument("--max-depth-cm", type=float, default=180.0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_fusion(parse_args())
