from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def _build_model(head_variant: str = "deep") -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    if head_variant == "compact":
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    else:
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
    model.eval()
    return model


def _load_model(model_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    deep_model = _build_model("deep").to(device)
    try:
        deep_model.load_state_dict(state_dict, strict=True)
        return deep_model
    except RuntimeError:
        compact_model = _build_model("compact").to(device)
        compact_model.load_state_dict(state_dict, strict=True)
        return compact_model


def _severity_stage(depth_cm: float) -> int:
    if depth_cm < 5:
        return 1
    if depth_cm < 20:
        return 2
    if depth_cm < 50:
        return 3
    if depth_cm < 80:
        return 4
    return 5


def _load_sample_paths(manifest_path: Path, sample_size: int) -> List[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    rows: List[Path] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rel = (row.get("image_path") or "").strip()
            if not rel:
                continue
            img = (repo_root / rel).resolve()
            if img.exists():
                rows.append(img)
    if sample_size > 0:
        rows = rows[:sample_size]
    return rows


def _predict_depths(model: nn.Module, image_paths: List[Path], device: torch.device) -> List[float]:
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    values: List[float] = []
    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            tensor = tfm(image).unsqueeze(0).to(device)
            normalized = float(model(tensor).squeeze().item())
            values.append(float(np.clip(normalized * 100.0, 0.0, 180.0)))
    return values


def _diff_stats(prod_depths: List[float], cand_depths: List[float]) -> Tuple[float, float]:
    deltas = [abs(a - b) for a, b in zip(prod_depths, cand_depths)]
    stage_drifts = [
        1 if _severity_stage(a) != _severity_stage(b) else 0
        for a, b in zip(prod_depths, cand_depths)
    ]
    mean_abs_delta = sum(deltas) / len(deltas) if deltas else 0.0
    stage_drift_rate = sum(stage_drifts) / len(stage_drifts) if stage_drifts else 0.0
    return mean_abs_delta, stage_drift_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow canary compare candidate checkpoint against production.")
    parser.add_argument("--production-model", default="models/best_flood_model_water_aware.pth")
    parser.add_argument("--candidate-model", default="models/candidate/best_flood_model_water_aware_candidate.pth")
    parser.add_argument("--manifest", default="test_images/evaluation_manifest_labeled.csv")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--max-mean-depth-delta-cm", type=float, default=15.0)
    parser.add_argument("--max-severity-drift-rate", type=float, default=0.15)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    prod_model_path = (repo_root / args.production_model).resolve()
    cand_model_path = (repo_root / args.candidate_model).resolve()
    manifest_path = (repo_root / args.manifest).resolve()

    if not cand_model_path.exists():
        raise SystemExit(
            f"Candidate checkpoint missing: {cand_model_path}. "
            "Generate it via scripts/retrain_flood_classifier.py before running shadow canary."
        )
    if not prod_model_path.exists():
        raise SystemExit(f"Production checkpoint missing: {prod_model_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest missing: {manifest_path}")

    samples = _load_sample_paths(manifest_path, args.sample_size)
    if not samples:
        raise SystemExit("No sample images available for canary comparison.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prod_model = _load_model(prod_model_path, device)
    cand_model = _load_model(cand_model_path, device)
    prod_depths = _predict_depths(prod_model, samples, device)
    cand_depths = _predict_depths(cand_model, samples, device)
    mean_abs_delta, stage_drift_rate = _diff_stats(prod_depths, cand_depths)

    print(
        {
            "sample_size": len(samples),
            "mean_abs_depth_delta_cm": round(mean_abs_delta, 4),
            "severity_stage_drift_rate": round(stage_drift_rate, 4),
            "max_mean_depth_delta_cm": args.max_mean_depth_delta_cm,
            "max_severity_drift_rate": args.max_severity_drift_rate,
        }
    )

    if mean_abs_delta > args.max_mean_depth_delta_cm:
        raise SystemExit(
            f"Canary failed: mean_abs_depth_delta_cm={mean_abs_delta:.4f} > {args.max_mean_depth_delta_cm:.4f}"
        )
    if stage_drift_rate > args.max_severity_drift_rate:
        raise SystemExit(
            f"Canary failed: severity_stage_drift_rate={stage_drift_rate:.4f} > {args.max_severity_drift_rate:.4f}"
        )


if __name__ == "__main__":
    main()
