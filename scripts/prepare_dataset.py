"""
Create a dataset manifest for flood depth training from a folder of unlabeled images.

This helper can:
- scan a source directory of images
- optionally copy them into flood_dataset/{split}
- write a `labels.csv` manifest with placeholder depth values

Usage:
    python scripts/prepare_dataset.py --source test_images --dataset-root flood_dataset --split train

If you already have labels, edit the generated CSV to fill in depth_cm values.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import List

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def scan_images(source_dir: Path) -> List[Path]:
    images: List[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
            images.append(path)
    return images


def create_manifest(manifest_path: Path, image_paths: List[Path], base_dir: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "depth_cm"])
        writer.writeheader()
        for path in image_paths:
            filename = str(path.relative_to(base_dir))
            writer.writerow({"filename": filename, "depth_cm": ""})


def prepare_dataset(source_dir: Path, dataset_root: Path, split: str, copy: bool) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    images = scan_images(source_dir)
    if not images:
        raise RuntimeError(f"No images found in {source_dir}")

    split_dir = dataset_root / split
    manifest_path = split_dir / "labels.csv"
    split_dir.mkdir(parents=True, exist_ok=True)

    if copy:
        copied_paths = []
        for path in images:
            target = split_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)
            copied_paths.append(target)
        create_manifest(manifest_path, copied_paths, split_dir)
    else:
        create_manifest(manifest_path, images, source_dir)

    print(f"Prepared {len(images)} images for split '{split}'")
    if copy:
        print(f"Copied images to: {split_dir}")
    else:
        print(f"Using original image files in: {source_dir}")
    print(f"Manifest created at: {manifest_path}")
    print("Edit depth_cm values in the manifest to add ground truth labels.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare flood dataset manifest for training")
    parser.add_argument("--source", type=Path, default=Path("test_images"), help="Source image folder")
    parser.add_argument("--dataset-root", type=Path, default=Path("flood_dataset"), help="Target dataset root")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name")
    parser.add_argument("--copy", action="store_true", help="Copy source images into the dataset split folder")

    args = parser.parse_args()
    prepare_dataset(args.source, args.dataset_root, args.split, args.copy)


if __name__ == "__main__":
    main()
