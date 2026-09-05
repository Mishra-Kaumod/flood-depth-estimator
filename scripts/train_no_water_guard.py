#new_script_23 Aug
"""Train the optional MobileNetV3-Small no-water classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

CLASS_NAMES = ["no_water", "water"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".avif"}


def build_model(pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    return model


class WaterPresenceDataset(Dataset):
    def __init__(self, data_dir: Path, transform: transforms.Compose):
        self.transform = transform
        self.items: list[tuple[Path, int]] = []
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_dir = data_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.items.append((image_path, class_index))
        if not self.items:
            raise RuntimeError(f"No images found under {data_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.items[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def split_indices(labels: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(set(labels)):
        class_indices = [index for index, item_label in enumerate(labels) if item_label == label]
        rng.shuffle(class_indices)
        val_count = max(1, round(len(class_indices) * val_fraction))
        if len(class_indices) > 1:
            val_count = min(val_count, len(class_indices) - 1)
        else:
            val_count = 0
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    source_dataset = WaterPresenceDataset(Path(args.data_dir), val_transform)
    labels = [label for _, label in source_dataset.items]
    train_indices, val_indices = split_indices(labels, args.val_fraction, args.seed)
    if not train_indices or not val_indices:
        raise ValueError("Each class needs enough images for a train/validation split")

    train_source = WaterPresenceDataset(Path(args.data_dir), train_transform)
    val_source = WaterPresenceDataset(Path(args.data_dir), val_transform)
    train_loader = DataLoader(Subset(train_source, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_source, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.pretrained).to(device)
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_accuracy = -1.0
    best_checkpoint: dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, device)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.3f} | val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.3f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_checkpoint = {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "class_names": CLASS_NAMES,
                "architecture": "mobilenet_v3_small",
                "epoch": epoch,
                "val_accuracy": float(val_accuracy),
                "val_loss": float(val_loss),
                "training_config": vars(args),
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, output_path)
    print(f"Saved checkpoint: {output_path}")
    print(json.dumps({"classes": CLASS_NAMES, "val_accuracy": best_accuracy, "device": str(device)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the optional no-water MobileNetV3-Small classifier")
    parser.add_argument("--data-dir", default="training_data/water_presence")
    parser.add_argument("--output", default="models/no_water_guard_mobilenet_v3_small.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
