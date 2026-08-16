#!/usr/bin/env python3
"""Train the EfficientNet candidate depth signal on prepared labeled splits."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class FloodDepthDataset(Dataset):
    def __init__(self, images_dir: Path, labels_csv: Path, max_depth_cm: float):
        self.images_dir = images_dir
        self.max_depth_cm = float(max_depth_cm)
        self.items: list[tuple[Path, float]] = []
        with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                filename = row.get("filename") or row.get("image_name")
                if not filename:
                    continue
                image_path = images_dir / filename
                if image_path.exists():
                    self.items.append((image_path, float(row["depth_cm"])))
        if not self.items:
            raise RuntimeError(f"No labeled images found in {images_dir} using {labels_csv}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path, depth_cm = self.items[idx]
        image = Image.open(image_path).convert("RGB")
        target = np.clip(depth_cm, 0.0, self.max_depth_cm) / self.max_depth_cm
        return {
            "image": self.transform(image),
            "target": torch.tensor(target, dtype=torch.float32),
            "depth_cm": torch.tensor(depth_cm, dtype=torch.float32),
        }


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


def load_matching_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    current = model.state_dict()
    matched = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    current.update(matched)
    model.load_state_dict(current)
    print(f"Loaded {len(matched)} / {len(current)} tensors from {checkpoint_path}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_depth_cm: float) -> tuple[float, float]:
    model.eval()
    criterion = nn.SmoothL1Loss(beta=0.05)
    total_loss = 0.0
    total_abs_cm = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            depth_cm = batch["depth_cm"].to(device).unsqueeze(1)
            pred = model(images)
            total_loss += criterion(pred, targets).item()
            total_abs_cm += torch.abs((pred * max_depth_cm) - depth_cm).sum().item()
            count += images.shape[0]
    return total_loss / max(1, len(loader)), total_abs_cm / max(1, count)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = build_model().to(device)
    if args.base_model:
        load_matching_checkpoint(model, Path(args.base_model), device)

    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    train_ds = FloodDepthDataset(Path(args.train_images), Path(args.train_labels), args.max_depth_cm)
    val_ds = FloodDepthDataset(Path(args.val_images), Path(args.val_labels), args.max_depth_cm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.05)
    best = {"val_mae_cm": float("inf")}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_mae = evaluate(model, val_loader, device, args.max_depth_cm)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss / max(1, len(train_loader)):.5f} | "
            f"val_loss={val_loss:.5f} | val_mae={val_mae:.2f}cm"
        )
        if val_mae < best["val_mae_cm"]:
            best = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "max_depth_cm": float(args.max_depth_cm),
                "val_loss": float(val_loss),
                "val_mae_cm": float(val_mae),
                "training_config": vars(args),
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, out_path)
    print(f"Saved best candidate to {out_path}")
    print(f"Best epoch={best['epoch']} val_mae={best['val_mae_cm']:.2f}cm max_depth_cm={best['max_depth_cm']:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", default="training_runs/20260816_154019/images/train")
    parser.add_argument("--train-labels", default="training_runs/20260816_154019/labels_train.csv")
    parser.add_argument("--val-images", default="training_runs/20260816_154019/images/val")
    parser.add_argument("--val-labels", default="training_runs/20260816_154019/labels_val.csv")
    parser.add_argument("--base-model", default="models/candidate/best_flood_model_water_aware.pth")
    parser.add_argument("--output", default="models/candidate/best_flood_model_water_aware.pth")
    parser.add_argument("--max-depth-cm", type=float, default=180.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())