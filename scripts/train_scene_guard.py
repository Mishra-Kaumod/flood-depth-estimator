#!/usr/bin/env python3
"""Train a three-class MobileNetV3 scene guard for flood decisions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms


CLASS_NAMES = ["no_water", "shallow_water", "meaningful_flood"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    return model


class SceneDataset(Dataset):
    def __init__(self, data_dir: Path, transform: transforms.Compose) -> None:
        self.transform = transform
        self.items: list[tuple[Path, int]] = []
        for index, class_name in enumerate(CLASS_NAMES):
            class_dir = data_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.items.append((path, index))
        if not self.items:
            raise RuntimeError(f"No images found under {data_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.items[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def stratified_split(labels: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in range(len(CLASS_NAMES)):
        indices = [idx for idx, item_label in enumerate(labels) if item_label == label]
        rng.shuffle(indices)
        val_count = max(1, round(len(indices) * val_fraction))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, list[float]]:
    model.eval()
    total_loss = 0.0
    totals = [0] * len(CLASS_NAMES)
    correct = [0] * len(CLASS_NAMES)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            for class_index in range(len(CLASS_NAMES)):
                mask = labels == class_index
                totals[class_index] += int(mask.sum().item())
                correct[class_index] += int((predictions[mask] == class_index).sum().item())
    recalls = [correct[idx] / max(1, totals[idx]) for idx in range(len(CLASS_NAMES))]
    return total_loss / max(1, sum(totals)), float(np.mean(recalls)), recalls


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    source = SceneDataset(Path(args.data_dir), eval_transform)
    labels = [label for _, label in source.items]
    train_indices, val_indices = stratified_split(labels, args.val_fraction, args.seed)
    train_source = SceneDataset(Path(args.data_dir), train_transform)
    val_source = SceneDataset(Path(args.data_dir), eval_transform)
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights /= class_weights.mean()
    train_labels = [labels[index] for index in train_indices]
    sample_weights = [float(class_weights[label]) for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_indices), replacement=True)
    train_loader = DataLoader(Subset(train_source, train_indices), batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(Subset(val_source, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best: dict[str, object] | None = None
    best_macro_recall = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss, macro_recall, recalls = evaluate(model, val_loader, criterion, device)
        recall_text = ", ".join(f"{name}={score:.3f}" for name, score in zip(CLASS_NAMES, recalls))
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss / max(1, len(train_loader)):.4f} | val_loss={val_loss:.4f} | macro_recall={macro_recall:.3f} | {recall_text}")
        if macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            best = {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "class_names": CLASS_NAMES,
                "architecture": "mobilenet_v3_small",
                "epoch": epoch,
                "val_macro_recall": macro_recall,
                "val_class_recall": dict(zip(CLASS_NAMES, recalls)),
                "val_loss": val_loss,
                "class_counts": dict(zip(CLASS_NAMES, class_counts.astype(int).tolist())),
                "training_config": vars(args),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, output)
    print(f"Saved checkpoint: {output}")
    print(json.dumps({"val_macro_recall": best_macro_recall, "device": str(device)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a three-class MobileNetV3 flood scene guard")
    parser.add_argument("--data-dir", default="training_runs/test0916_scene_guard/data")
    parser.add_argument("--output", default="models/candidate/scene_guard_3class_test0916.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
