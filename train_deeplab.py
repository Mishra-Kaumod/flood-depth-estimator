import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 on FloodNet binary water segmentation")
    parser.add_argument("--data-dir", type=str, default="data/floodnet", help="Root dataset directory")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--output-dir", type=str, default="training/checkpoints", help="Checkpoint output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


class FloodNetDataset(data.Dataset):
    def __init__(self, images_dir, masks_dir, image_size=512, augment=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment
        self.samples = self._collect_samples()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _collect_samples(self):
        image_files = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        samples = []
        for img_path in image_files:
            mask_path = self.masks_dir / (img_path.stem + ".png")
            if mask_path.exists():
                samples.append((img_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._random_augment(image, mask)
        else:
            image = F.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        tensor = F.to_tensor(image)
        tensor = self.normalize(tensor)

        target = torch.from_numpy(np.array(mask, dtype=np.uint8))
        target = (target > 0).long()

        return tensor, target

    def _random_augment(self, image, mask):
        # Resize before crop to preserve aspect ratio for small images
        image = F.resize(image, [self.image_size + 32, self.image_size + 32], interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.image_size + 32, self.image_size + 32], interpolation=InterpolationMode.NEAREST)

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.image_size, self.image_size))
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.05, 0.05)
            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)
            image = F.adjust_saturation(image, saturation)
            image = F.adjust_hue(image, hue)

        if random.random() > 0.5:
            image = F.gaussian_blur(image, kernel_size=3)

        return image, mask


def create_dataloader(root_dir, split, batch_size, num_workers, image_size, augment=False):
    images_dir = Path(root_dir) / split / "images"
    masks_dir = Path(root_dir) / split / "masks"
    dataset = FloodNetDataset(images_dir, masks_dir, image_size=image_size, augment=augment)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, pin_memory=True)


def build_model(device):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights, progress=True)
    model.classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)
    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        model.aux_classifier = None
    return model.to(device)


def calculate_metrics(pred_mask, target_mask, eps=1e-8):
    pred = pred_mask.view(-1)
    target = target_mask.view(-1)

    tp = int(((pred == 1) & (target == 1)).sum().item())
    fp = int(((pred == 1) & (target == 0)).sum().item())
    fn = int(((pred == 0) & (target == 1)).sum().item())
    tn = int(((pred == 0) & (target == 0)).sum().item())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, iou, f1, tp, fp, fn, tn


def train_one_epoch(model, dataloader, optimizer, scaler, device, criterion):
    model.train()
    running_loss = 0.0
    metrics = np.zeros(4, dtype=np.float64)
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)["out"]
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        for pred, target in zip(preds, targets):
            p, r, i, f1, *_ = calculate_metrics(pred, target)
            metrics += np.array([p, r, i, f1], dtype=np.float64)
            total += 1

    avg_loss = running_loss / max(total, 1)
    avg_metrics = metrics / max(total, 1)
    return avg_loss, avg_metrics.tolist()


def evaluate(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    metrics = np.zeros(4, dtype=np.float64)
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)["out"]
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            for pred, target in zip(preds, targets):
                p, r, i, f1, *_ = calculate_metrics(pred, target)
                metrics += np.array([p, r, i, f1], dtype=np.float64)
                total += 1

    avg_loss = running_loss / max(total, 1)
    avg_metrics = metrics / max(total, 1)
    return avg_loss, avg_metrics.tolist()


def write_csv_header(csv_path):
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "train_precision",
                "train_recall",
                "train_iou",
                "train_f1",
                "val_precision",
                "val_recall",
                "val_iou",
                "val_f1",
                "lr",
                "elapsed_seconds",
            ])


def append_csv_row(csv_path, row):
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"
    csv_path = output_dir / "training_metrics.csv"

    train_loader = create_dataloader(args.data_dir, "train", args.batch_size, args.num_workers, 512, augment=True)
    val_loader = create_dataloader(args.data_dir, "val", args.batch_size, args.num_workers, 512, augment=False)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    write_csv_header(csv_path)

    best_iou = 0.0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
        val_loss, val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        train_precision, train_recall, train_iou, train_f1 = train_metrics
        val_precision, val_recall, val_iou, val_f1 = val_metrics
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        append_csv_row(csv_path, [
            epoch,
            train_loss,
            val_loss,
            train_precision,
            train_recall,
            train_iou,
            train_f1,
            val_precision,
            val_recall,
            val_iou,
            val_f1,
            lr,
            round(elapsed, 2),
        ])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_iou={train_iou:.4f} val_iou={val_iou:.4f} | "
            f"val_prec={val_precision:.4f} val_recall={val_recall:.4f} val_f1={val_f1:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"  → New best model saved to {best_path} (val_iou={best_iou:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement}/{args.patience} epochs")

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered")
            break

    total_elapsed = time.time() - start_time
    print(f"Training complete. Best val_iou={best_iou:.4f}. Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
