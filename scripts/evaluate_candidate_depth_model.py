#!/usr/bin/env python3
"""Evaluate production and candidate EfficientNet depth checkpoints.

This script is intentionally narrow: it compares two checkpoints on a labeled
CSV split and writes per-image errors so demo images can be chosen from known
behavior instead of gut feel.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    return model


def load_model(path: Path, device: torch.device, default_max_depth_cm: float) -> tuple[nn.Module, float]:
    model = build_model().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    max_depth_cm = float(checkpoint.get("max_depth_cm", default_max_depth_cm)) if isinstance(checkpoint, dict) else default_max_depth_cm
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, max_depth_cm


def predict_cm(model: nn.Module, image_path: Path, transform, device: torch.device, max_depth_cm: float) -> float:
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        return float(model(tensor).squeeze().item()) * max_depth_cm


def summarize(errors: list[float]) -> dict[str, float]:
    ordered = sorted(errors)
    n = len(ordered)
    p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
    return {
        "mae_cm": sum(ordered) / n,
        "median_error_cm": ordered[n // 2],
        "p95_error_cm": ordered[p95_idx],
        "max_error_cm": max(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="training_runs/20260816_154019/images/test")
    parser.add_argument("--labels", default="training_runs/20260816_154019/labels_test.csv")
    parser.add_argument("--production-model", default="models/best_flood_model_water_aware.pth")
    parser.add_argument("--candidate-model", default="models/candidate/best_flood_model_water_aware.pth")
    parser.add_argument("--out", default="reports/candidate_depth_eval.csv")
    parser.add_argument("--production-max-depth-cm", type=float, default=100.0)
    parser.add_argument("--candidate-max-depth-cm", type=float, default=180.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images_dir = Path(args.images_dir)
    with open(args.labels, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    production, production_max_depth_cm = load_model(Path(args.production_model), device, args.production_max_depth_cm)
    candidate, candidate_max_depth_cm = load_model(Path(args.candidate_model), device, args.candidate_max_depth_cm)

    out_rows = []
    prod_errors: list[float] = []
    cand_errors: list[float] = []
    for row in rows:
        filename = row["filename"]
        actual = float(row["depth_cm"])
        image_path = images_dir / filename
        prod_pred = predict_cm(production, image_path, transform, device, production_max_depth_cm)
        cand_pred = predict_cm(candidate, image_path, transform, device, candidate_max_depth_cm)
        prod_error = abs(prod_pred - actual)
        cand_error = abs(cand_pred - actual)
        prod_errors.append(prod_error)
        cand_errors.append(cand_error)
        out_rows.append(
            {
                "filename": filename,
                "actual_depth_cm": f"{actual:.2f}",
                "production_pred_cm": f"{prod_pred:.2f}",
                "production_error_cm": f"{prod_error:.2f}",
                "candidate_pred_cm": f"{cand_pred:.2f}",
                "candidate_error_cm": f"{cand_error:.2f}",
                "candidate_better": cand_error < prod_error,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    worst = sorted(out_rows, key=lambda item: float(item["candidate_error_cm"]), reverse=True)[:10]
    print(f"Wrote {out_path}")
    print("Production:", {k: round(v, 3) for k, v in summarize(prod_errors).items()})
    print("Candidate:", {k: round(v, 3) for k, v in summarize(cand_errors).items()})
    print("Candidate better on", sum(1 for r in out_rows if r["candidate_better"]), "of", len(out_rows), "images")
    print("Worst candidate errors:")
    for row in worst:
        print(
            f"  {row['filename']}: actual={row['actual_depth_cm']} "
            f"candidate={row['candidate_pred_cm']} error={row['candidate_error_cm']}"
        )


if __name__ == "__main__":
    main()
