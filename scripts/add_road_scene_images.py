#!/usr/bin/env python3
"""Import labelled road-scene images into the fixed four-class dataset splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = ("dry_road", "wet_road", "shallow_flood", "meaningful_flood")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    return [path for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def target_splits(count: int, seed: int) -> list[str]:
    values = ["train"] * round(count * 0.70)
    values += ["validation"] * round(count * 0.15)
    values += ["test"] * (count - len(values))
    random.Random(seed).shuffle(values)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Add labelled road-scene images without duplicate leakage")
    parser.add_argument("--source-dir", required=True, help="Contains dry_road/, wet_road/, shallow_flood/, meaningful_flood/")
    parser.add_argument("--dataset-dir", default="training_data/road_scene_classifier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    dataset_dir = Path(args.dataset_dir)
    existing_hashes = {sha256(path) for path in image_files(dataset_dir)}
    manifest_path = dataset_dir / "scene_classifier_manifest.csv"
    rows: list[dict[str, str]] = []

    for class_name in CLASS_NAMES:
        class_source = source_dir / class_name
        if not class_source.exists():
            continue
        candidates = image_files(class_source)
        unique: list[Path] = []
        batch_hashes: set[str] = set()
        for path in candidates:
            digest = sha256(path)
            if digest not in existing_hashes and digest not in batch_hashes:
                unique.append(path)
                batch_hashes.add(digest)
        for index, (path, split) in enumerate(zip(unique, target_splits(len(unique), args.seed)), start=1):
            destination_dir = dataset_dir / split / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"incoming_{index:04d}_{path.name}"
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(path, destination)
            rows.append({
                "split": split,
                "scene_class": class_name,
                "source_path": str(path.resolve()),
                "dataset_path": str(destination.relative_to(dataset_dir)),
            })
        print(f"{class_name}: added={len(unique)}, skipped_duplicates={len(candidates) - len(unique)}")

    if rows:
        with manifest_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "scene_class", "source_path", "dataset_path"])
            writer.writerows(rows)
    totals = Counter()
    for path in image_files(dataset_dir):
        parts = path.relative_to(dataset_dir).parts
        if len(parts) >= 2 and parts[0] in {"train", "validation", "test"}:
            totals[(parts[0], parts[1])] += 1
    for split in ("train", "validation", "test"):
        print(split, {name: totals[(split, name)] for name in CLASS_NAMES})


if __name__ == "__main__":
    main()
