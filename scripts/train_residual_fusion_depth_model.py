#!/usr/bin/env python3
"""Train a residual fusion model anchored on EfficientNet depth.

EfficientNet predicts the starting depth. This model learns a bounded correction
from the rest of the pipeline signals, so noisy water/reference/depth features
cannot wildly override the trained image model.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.train_fusion_depth_model import FEATURE_NAMES, load_or_build_features, mae


class ResidualFusionDepthModel(nn.Module):
    def __init__(self, input_dim: int, max_residual_cm: float):
        super().__init__()
        self.max_residual_cm = float(max_residual_cm)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, base_depth_cm: torch.Tensor) -> torch.Tensor:
        residual_cm = self.net(x) * self.max_residual_cm
        return torch.clamp(base_depth_cm + residual_cm, min=0.0, max=180.0)


def make_tensors(frame: pd.DataFrame, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    x = frame[FEATURE_NAMES].astype(float).to_numpy(dtype=np.float32)
    y = frame["actual_depth_cm"].astype(float).to_numpy(dtype=np.float32).reshape(-1, 1)
    base = frame["efficientnet_candidate_depth_cm"].astype(float).to_numpy(dtype=np.float32).reshape(-1, 1)
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x = (x - mean) / std
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(base, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        mean,
        std,
    )


def predict_cm(model: nn.Module, x: torch.Tensor, base: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            xb = x[start : start + 256].to(device)
            bb = base[start : start + 256].to(device)
            out.append(model(xb, bb).cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = load_or_build_features(args)
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()

    x_train, base_train, y_train, mean, std = make_tensors(train_df)
    x_val, base_val, y_val, _, _ = make_tensors(val_df, mean, std)
    x_test, base_test, y_test, _, _ = make_tensors(test_df, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = ResidualFusionDepthModel(len(FEATURE_NAMES), args.max_residual_cm).to(device)
    loader = DataLoader(TensorDataset(x_train, base_train, y_train), batch_size=args.batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=args.huber_beta_cm)

    best_state: dict[str, Any] | None = None
    best_val_mae = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, bb, yb in loader:
            xb = xb.to(device)
            bb = bb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, bb)
            residual = pred - bb
            loss = criterion(pred, yb) + args.residual_penalty * torch.mean(torch.abs(residual))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_pred = predict_cm(model, x_val, base_val, device)
        val_actual = val_df["actual_depth_cm"].astype(float).to_numpy()
        val_mae = mae(val_actual, val_pred)
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss / max(1, len(loader)):.5f} val_mae={val_mae:.2f}cm", flush=True)

        if val_mae + 0.01 < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch}; best_val_mae={best_val_mae:.2f}cm", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model.")
    model.load_state_dict(best_state)

    for name, split_df, x, base in [
        ("train", train_df, x_train, base_train),
        ("val", val_df, x_val, base_val),
        ("test", test_df, x_test, base_test),
    ]:
        pred = predict_cm(model, x, base, device)
        actual = split_df["actual_depth_cm"].astype(float).to_numpy()
        print(
            f"{name} MAE | residual_fusion={mae(actual, pred):.2f}cm | "
            f"current_pipeline={mae(actual, split_df['pipeline_depth_cm'].astype(float).to_numpy()):.2f}cm | "
            f"efficientnet={mae(actual, split_df['efficientnet_candidate_depth_cm'].astype(float).to_numpy()):.2f}cm",
            flush=True,
        )

    test_pred = predict_cm(model, x_test, base_test, device)
    report = test_df.copy()
    report["residual_fusion_pred_depth_cm"] = np.round(test_pred, 2)
    report["residual_fusion_abs_error_cm"] = np.round(np.abs(report["actual_depth_cm"].astype(float).to_numpy() - test_pred), 2)
    report["pipeline_abs_error_cm"] = np.round(np.abs(report["actual_depth_cm"].astype(float) - report["pipeline_depth_cm"].astype(float)), 2)
    report["efficientnet_abs_error_cm"] = np.round(np.abs(report["actual_depth_cm"].astype(float) - report["efficientnet_candidate_depth_cm"].astype(float)), 2)

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    checkpoint = {
        "architecture": "residual_fusion_depth_v1",
        "model_state_dict": best_state,
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean.astype(float).tolist(),
        "feature_std": std.astype(float).tolist(),
        "max_depth_cm": 180.0,
        "max_residual_cm": float(args.max_residual_cm),
        "best_val_mae_cm": float(best_val_mae),
        "training_config": vars(args),
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)

    print(f"Saved residual fusion model: {model_path}", flush=True)
    print(f"Saved test report: {report_path}", flush=True)

    worst = report.sort_values("residual_fusion_abs_error_cm", ascending=False).head(10)
    print("Worst residual fusion test errors:", flush=True)
    for _, row in worst.iterrows():
        print(
            f"  {row['filename']}: actual={float(row['actual_depth_cm']):.2f}cm "
            f"fusion={float(row['residual_fusion_pred_depth_cm']):.2f}cm "
            f"pipeline={float(row['pipeline_depth_cm']):.2f}cm "
            f"efficientnet={float(row['efficientnet_candidate_depth_cm']):.2f}cm",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual fusion model anchored on EfficientNet")
    parser.add_argument("--split-dir", default="training_runs/20260816_154019")
    parser.add_argument("--features-out", default="reports/fusion_training_features_sample.csv")
    parser.add_argument("--model-out", default="models/candidate/residual_fusion_depth_model.pt")
    parser.add_argument("--report-out", default="reports/residual_fusion_depth_eval.csv")
    parser.add_argument("--max-residual-cm", type=float, default=35.0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--huber-beta-cm", type=float, default=8.0)
    parser.add_argument("--residual-penalty", type=float, default=0.002)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())