"""
Auto-fill depth labels for a dataset using the current flood depth prediction pipeline.

This script is intended to help you bootstrap training labels by using the current model
as a suggestion engine. It writes a CSV manifest where blank values are replaced with
predicted depths.

Usage:
    python scripts/auto_fill_labels.py --dataset-dir flood_dataset/train

Once generated, review the file and correct any values before training.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segformer_yolo_depthv2_pipeline import get_segformer_yolo_depthv2_pipeline


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not manifest_path.exists():
        return labels

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            depth = (row.get("depth_cm") or "").strip()
            if filename:
                labels[filename] = depth
    return labels


def scan_images(dataset_dir: Path) -> List[Path]:
    images: List[Path] = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
            if path.name.lower() == "labels.csv":
                continue
            images.append(path)
    return images


def write_manifest(manifest_path: Path, rows: List[Tuple[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "depth_cm"])
        writer.writeheader()
        for filename, depth in rows:
            writer.writerow({"filename": filename, "depth_cm": depth})


def predict_depths(dataset_dir: Path, manifest_path: Path, preserve_existing: bool = True) -> None:
    pipeline = get_segformer_yolo_depthv2_pipeline()
    existing_labels = load_manifest(manifest_path)
    images = scan_images(dataset_dir)

    rows: List[Tuple[str, str]] = []
    for img_path in images:
        filename = img_path.name
        current_label = existing_labels.get(filename, "")
        if preserve_existing and current_label.strip():
            rows.append((filename, current_label.strip()))
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"WARNING: could not read image {img_path}")
            rows.append((filename, ""))
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            result = pipeline.predict(image_rgb)
            depth_cm = float(result.get("depth_cm", 0.0))
            rows.append((filename, f"{depth_cm:.2f}"))
        except Exception as exc:
            print(f"ERROR: prediction failed for {filename}: {exc}")
            rows.append((filename, ""))

    write_manifest(manifest_path, rows)
    print(f"Wrote {len(rows)} label rows to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-fill depth labels using current pipeline predictions")
    parser.add_argument("--dataset-dir", type=Path, default=Path("flood_dataset/train"), help="Dataset directory containing images and labels.csv")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to manifest file (defaults to labels.csv in dataset dir)")
    parser.add_argument("--preserve-existing", action="store_true", default=True, help="Keep any non-empty depth values in the existing manifest")

    args = parser.parse_args()
    manifest_path = args.manifest or (args.dataset_dir / "labels.csv")

    if not args.dataset_dir.exists() or not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")

    predict_depths(args.dataset_dir, manifest_path, preserve_existing=args.preserve_existing)


if __name__ == "__main__":
    main()
