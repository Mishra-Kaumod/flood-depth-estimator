#!/usr/bin/env python3
"""Train and evaluate the four-class road-scene classifier on fixed data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


CLASS_NAMES = ["dry_road", "wet_road", "shallow_flood", "meaningful_flood"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    return model


class RoadSceneDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose, blocked_hashes: set[str] | None = None) -> None:
        self.transform = transform
        self.items: list[tuple[Path, int, str]] = []
        self.skipped_duplicates: list[Path] = []
        seen_hashes: set[str] = set()
        blocked_hashes = blocked_hashes or set()
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                digest = file_hash(path)
                if digest in blocked_hashes or digest in seen_hashes:
                    self.skipped_duplicates.append(path)
                    continue
                seen_hashes.add(digest)
                self.items.append((path, label, digest))
        if not self.items:
            raise RuntimeError(f"No usable images found under {root}")

    @property
    def hashes(self) -> set[str]:
        return {digest for _, _, digest in self.items}

    @property
    def labels(self) -> list[int]:
        return [label for _, label, _ in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label, _ = self.items[index]
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB")), torch.tensor(label, dtype=torch.long)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, object]:
    model.eval()
    loss_total = 0.0
    total = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    confidences: list[float] = []
    margins: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += criterion(logits, labels).item() * labels.size(0)
            total += labels.size(0)
            probabilities = torch.softmax(logits, dim=1)
            top_values = torch.topk(probabilities, k=min(2, len(CLASS_NAMES)), dim=1).values
            confidences.extend(top_values[:, 0].cpu().tolist())
            margins.extend((top_values[:, 0] - top_values[:, 1]).cpu().tolist())
            predictions = logits.argmax(dim=1)
            for actual, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                confusion[actual, predicted] += 1
    recalls = np.divide(np.diag(confusion), np.maximum(1, confusion.sum(axis=1)))
    precisions = np.divide(np.diag(confusion), np.maximum(1, confusion.sum(axis=0)))
    return {
        "loss": loss_total / max(1, total),
        "accuracy": float(np.trace(confusion) / max(1, total)),
        "macro_recall": float(recalls.mean()),
        "class_recall": dict(zip(CLASS_NAMES, recalls.tolist())),
        "class_precision": dict(zip(CLASS_NAMES, precisions.tolist())),
        "confusion_matrix": confusion.tolist(),
        "confidence_summary": {
            "mean": float(np.mean(confidences)) if confidences else 0.0,
            "p10": float(np.percentile(confidences, 10)) if confidences else 0.0,
            "p50": float(np.percentile(confidences, 50)) if confidences else 0.0,
            "p90": float(np.percentile(confidences, 90)) if confidences else 0.0,
        },
        "margin_summary": {
            "mean": float(np.mean(margins)) if margins else 0.0,
            "p10": float(np.percentile(margins, 10)) if margins else 0.0,
            "p50": float(np.percentile(margins, 50)) if margins else 0.0,
            "p90": float(np.percentile(margins, 90)) if margins else 0.0,
        },
        "samples": total,
    }


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    return train_transform, eval_transform


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    train_transform, eval_transform = build_transforms()
    train_data = RoadSceneDataset(data_dir / "train", train_transform)
    validation_data = RoadSceneDataset(data_dir / "validation", eval_transform, train_data.hashes)
    test_data = RoadSceneDataset(data_dir / "test", eval_transform, train_data.hashes | validation_data.hashes)

    counts = np.bincount(train_data.labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights /= class_weights.mean()
    sample_weights = [float(class_weights[label]) for label in train_data.labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_data), replacement=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(json.dumps({
        "device": str(device),
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "test_samples": len(test_data),
        "excluded_validation_duplicates": [str(path) for path in validation_data.skipped_duplicates],
        "excluded_test_duplicates": [str(path) for path in test_data.skipped_duplicates],
        "train_class_counts": dict(zip(CLASS_NAMES, counts.astype(int).tolist())),
    }, indent=2))

    model = build_model(args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state: dict[str, object] | None = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        validation = evaluate(model, validation_loader, criterion, device)
        recall = ", ".join(f"{name}={score:.3f}" for name, score in validation["class_recall"].items())
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss / max(1, len(train_loader)):.4f} | val_loss={validation['loss']:.4f} | val_macro_recall={validation['macro_recall']:.3f} | {recall}")
        if validation["macro_recall"] > best_score:
            best_score = float(validation["macro_recall"])
            best_state = {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "class_names": CLASS_NAMES,
                "architecture": "mobilenet_v3_small",
                "epoch": epoch,
                "validation": validation,
                "train_class_counts": dict(zip(CLASS_NAMES, counts.astype(int).tolist())),
                "training_config": vars(args),
            }
    assert best_state is not None
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    best_state["test"] = test_metrics
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"validation": best_state["validation"], "test": test_metrics}, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(output), "metrics": str(metrics_path), "best_epoch": best_state["epoch"], "test": test_metrics}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a four-class MobileNetV3 road-scene classifier")
    parser.add_argument("--data-dir", default="training_data/scene_classifier")
    parser.add_argument("--output", default="models/candidate/road_scene_classifier_4class_scene_guard.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
