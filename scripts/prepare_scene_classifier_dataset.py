#!/usr/bin/env python3
"""Prepare a verified, hash-deduplicated four-class scene dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def image_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def read_verified_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Verified source manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scene_class", "dataset_path"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError(f"{path} must contain scene_class and dataset_path columns")
        rows: dict[str, str] = {}
        for row in reader:
            scene_class = (row.get("scene_class") or "").strip()
            dataset_path = (row.get("dataset_path") or "").strip()
            if scene_class in CLASS_NAMES and dataset_path:
                rows[dataset_path.replace("\\", "/")] = scene_class
        return rows


def resolve_dataset_path(repo_root: Path, dataset_path: str, local_by_name: dict[str, list[Path]]) -> Path | None:
    candidate = Path(dataset_path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        matches = local_by_name.get(candidate.name.lower(), [])
        return matches[0] if len(matches) == 1 else None
    normalized = dataset_path.replace("\\", "/")
    prefix = "training_data/road_scene_classifier/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    candidate = repo_root / "training_data" / "road_scene_classifier" / normalized
    if candidate.exists():
        return candidate
    matches = local_by_name.get(candidate.name.lower(), [])
    return matches[0] if len(matches) == 1 else None


def target_splits(count: int, seed: int) -> list[str]:
    test_count = max(1, int(count * 0.30 + 0.999999))
    remaining = count - test_count
    validation_count = int(remaining * 0.15 + 0.5) if remaining else 0
    values = ["test"] * test_count
    values += ["validation"] * validation_count
    values += ["train"] * (count - len(values))
    random.Random(seed).shuffle(values)
    return values


def collect_sources(repo_root: Path, source_manifest: Path) -> tuple[list[tuple[Path, str, str]], list[str]]:
    manifest_rows = read_verified_manifest(source_manifest)
    local_root = repo_root / "training_data" / "road_scene_classifier"
    local_by_name: dict[str, list[Path]] = {}
    for path in image_files(local_root):
        local_by_name.setdefault(path.name.lower(), []).append(path)
    sources: list[tuple[Path, str, str]] = []
    missing: list[str] = []
    for dataset_path, scene_class in manifest_rows.items():
        source = resolve_dataset_path(repo_root, dataset_path, local_by_name)
        if source is None:
            missing.append(dataset_path)
            continue
        sources.append((source, scene_class, dataset_path))
    return sources, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default="training_data/road_scene_classifier/scene_classifier_manifest.csv")
    parser.add_argument("--output-dir", default="training_data/scene_classifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-dry", type=int, default=80)
    parser.add_argument("--min-wet", type=int, default=100)
    parser.add_argument("--min-shallow", type=int, default=100)
    parser.add_argument("--min-meaningful", type=int, default=100)
    parser.add_argument("--allow-incomplete", action="store_true", help="Build available data and return success despite target shortfalls")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_manifest = repo_root / args.source_manifest
    output_dir = repo_root / args.output_dir
    sources, missing = collect_sources(repo_root, source_manifest)

    unique: dict[str, tuple[Path, str, str]] = {}
    duplicate_count = 0
    for source, scene_class, dataset_path in sources:
        digest = sha256(source)
        if digest in unique:
            duplicate_count += 1
            continue
        unique[digest] = (source, scene_class, dataset_path)

    by_class: dict[str, list[tuple[str, Path, str]]] = {name: [] for name in CLASS_NAMES}
    for digest, (source, scene_class, dataset_path) in unique.items():
        by_class[scene_class].append((digest, source, dataset_path))
    for class_name in CLASS_NAMES:
        by_class[class_name].sort(key=lambda item: item[0])

    if output_dir.exists():
        for path in image_files(output_dir):
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    split_counts: Counter[tuple[str, str]] = Counter()

    for class_name in CLASS_NAMES:
        entries = by_class[class_name]
        splits = target_splits(len(entries), args.seed)
        for index, ((digest, source, dataset_path), split) in enumerate(zip(entries, splits), start=1):
            destination_dir = output_dir / split / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{index:04d}_{source.name}"
            shutil.copy2(source, destination)
            split_counts[(split, class_name)] += 1
            manifest_rows.append(
                {
                    "split": split,
                    "scene_class": class_name,
                    "sha256": digest,
                    "source_path": str(source.resolve()),
                    "source_manifest_path": dataset_path,
                    "dataset_path": str(destination.relative_to(repo_root)).replace("\\", "/"),
                }
            )

    manifest_path = output_dir / "scene_classifier_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["split", "scene_class", "sha256", "source_path", "source_manifest_path", "dataset_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    minimums = dict(zip(CLASS_NAMES, (args.min_dry, args.min_wet, args.min_shallow, args.min_meaningful)))
    totals = {class_name: len(by_class[class_name]) for class_name in CLASS_NAMES}
    shortfalls = {name: minimum - totals[name] for name, minimum in minimums.items() if totals[name] < minimum}
    report = {
        "source_manifest": str(source_manifest),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "total_counts": totals,
        "split_counts": {
            split: {name: split_counts[(split, name)] for name in CLASS_NAMES}
            for split in ("train", "validation", "test")
        },
        "test_fraction": {name: split_counts[("test", name)] / max(1, totals[name]) for name in CLASS_NAMES},
        "duplicate_count": duplicate_count,
        "missing_manifest_paths": missing,
        "shortfalls": shortfalls,
        "named_wet_road_files_present": {
            filename: any(
                Path(item[1]).stem.lower().endswith(Path(filename).stem.lower())
                and Path(item[1]).suffix.lower() == Path(filename).suffix.lower()
                for item in by_class["wet_road"]
            )
            for filename in ("wet road _11.jpg", "wet road _12.jpg")
        },
    }
    report_path = output_dir / "dataset_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if shortfalls and not args.allow_incomplete:
        print("Dataset targets are incomplete; add verified images before training.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())