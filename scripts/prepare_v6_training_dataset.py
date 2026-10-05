#!/usr/bin/env python3
"""Prepare future V6 data without invoking inference or training.

Inputs are an explicit new-image manifest and, optionally, a directory containing
those images. No project test/challenge folders are discovered or enumerated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from src.v6_training_data_contract import (
    MANIFEST_COLUMNS, ManifestRow, manifest_hash, quarantine_rows, sha256_file,
    freeze_challenge, write_manifest,
)


class UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8)), dtype=np.uint8)
    return int("".join("1" if value else "0" for value in (pixels[:, 1:] > pixels[:, :-1]).ravel()), 2)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def read_raw(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError("manifest header does not match v6_new_image_manifest.csv")
        return [dict(row) for row in reader]


def stable_score(seed: int, value: str) -> int:
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(), 16)


def split_groups(rows: list[ManifestRow], seed: int, validation_fraction: float, challenge_fraction: float) -> dict[str, str]:
    if not 0 <= validation_fraction < 1 or not 0 <= challenge_fraction < 1 or validation_fraction + challenge_fraction >= 1:
        raise ValueError("validation/challenge fractions must be non-negative and sum to less than one")
    groups: dict[str, list[ManifestRow]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    ordered = sorted(groups, key=lambda group: stable_score(seed, group))
    total = len(rows)
    targets = {"CHALLENGE": round(total * challenge_fraction), "VALIDATION": round(total * validation_fraction)}
    assigned: dict[str, str] = {}
    counts = {"CHALLENGE": 0, "VALIDATION": 0}
    for group in ordered:
        size = len(groups[group])
        for split in ("CHALLENGE", "VALIDATION"):
            if counts[split] < targets[split] and abs((counts[split] + size) - targets[split]) <= abs(counts[split] - targets[split]):
                assigned[group] = split
                counts[split] += size
                break
        else:
            assigned[group] = "TRAIN"
    return assigned


def summarize(rows: list[ManifestRow]) -> dict[str, Any]:
    def count(field: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for row in rows:
            raw_value = getattr(row, field)
            value = str(getattr(raw_value, "value", raw_value))
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))
    from src.v6_training_data_contract import depth_bucket
    buckets: dict[str, int] = {}
    for row in rows:
        bucket = depth_bucket(row.depth_cm)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return {
        "rows": len(rows), "groups": len({row.group_id for row in rows}),
        "scene_type": count("scene_type"), "depth_bucket": dict(sorted(buckets.items())),
        "label_confidence": count("label_confidence"), "measurement_source": count("measurement_source"),
        "source_session_id": count("source_session_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic V6 training-readiness manifests")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", help="Optional explicit new-image directory; omit to record visual checks as not_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--challenge-fraction", type=float, default=0.15)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    rows, quarantined = quarantine_rows(read_raw(Path(args.manifest)))
    image_root = Path(args.image_root) if args.image_root else None
    enriched: list[ManifestRow] = []
    image_hashes: dict[str, str] = {}
    visual_hashes: dict[str, int] = {}
    for row in rows:
        image_path = image_root / row.filename if image_root else None
        if image_path is None or not image_path.is_file():
            enriched.append(replace(row, visual_near_duplicate_status="not_run"))
            continue
        computed = sha256_file(image_path)
        if row.sha256 and row.sha256 != computed:
            item = row.to_mapping(); item["quarantine_reason"] = "sha256_mismatch"
            quarantined.append(item); continue
        image_hashes[row.image_id] = computed
        visual_hashes[row.image_id] = dhash(image_path)
        enriched.append(replace(row, sha256=computed, visual_near_duplicate_status="no_match"))
    rows = enriched
    union = UnionFind(row.image_id for row in rows)
    audit: list[dict[str, Any]] = []
    by_sha: dict[str, list[str]] = {}
    by_session: dict[str, list[str]] = {}
    for row in rows:
        if row.sha256:
            by_sha.setdefault(row.sha256, []).append(row.image_id)
        by_session.setdefault(row.source_session_id, []).append(row.image_id)
    for sha, ids in by_sha.items():
        for item in ids[1:]: union.union(ids[0], item)
        if len(ids) > 1: audit.append({"kind": "exact_sha256_duplicate", "members": ids, "evidence": sha})
    for session, ids in by_session.items():
        for item in ids[1:]: union.union(ids[0], item)
        if len(ids) > 1: audit.append({"kind": "source_session_group", "members": ids, "evidence": session})
    ids = sorted(visual_hashes)
    near_members: set[str] = set()
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            distance = hamming(visual_hashes[left], visual_hashes[right])
            if distance <= args.near_duplicate_hamming:
                union.union(left, right); near_members.update((left, right))
                audit.append({"kind": "visual_near_duplicate", "members": [left, right], "evidence": {"dhash_distance": distance}})
    grouped = []
    for row in rows:
        status = "near_duplicate" if row.image_id in near_members else row.visual_near_duplicate_status
        grouped.append(replace(row, group_id=f"v6grp:{union.find(row.image_id)}", visual_near_duplicate_status=status))
    assignments = split_groups(grouped, args.seed, args.validation_fraction, args.challenge_fraction)
    split_rows = {"TRAIN": [], "VALIDATION": [], "CHALLENGE": []}
    for row in grouped:
        split_rows[assignments[row.group_id]].append(row)
    for split, output_name in (("TRAIN", "train_manifest.csv"), ("VALIDATION", "validation_manifest.csv"), ("CHALLENGE", "challenge_manifest.csv")):
        write_manifest(output / output_name, split_rows[split])
    leakage = {
        "exact_sha256_overlap": {}, "group_overlap": {}, "visual_near_duplicate_status": "not_run" if not image_root else "run",
    }
    for left, right in (("TRAIN", "VALIDATION"), ("TRAIN", "CHALLENGE"), ("VALIDATION", "CHALLENGE")):
        pair = f"{left}-{right}"
        leakage["exact_sha256_overlap"][pair] = sorted({r.sha256 for r in split_rows[left] if r.sha256} & {r.sha256 for r in split_rows[right] if r.sha256})
        leakage["group_overlap"][pair] = sorted({r.group_id for r in split_rows[left]} & {r.group_id for r in split_rows[right]})
    config = {"seed": args.seed, "validation_fraction": args.validation_fraction, "challenge_fraction": args.challenge_fraction, "near_duplicate_hamming": args.near_duplicate_hamming}
    leakage["passed"] = all(not values for source in ("exact_sha256_overlap", "group_overlap") for values in leakage[source].values())
    (output / "duplicate_and_group_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "leakage_check.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    (output / "split_config.json").write_text(json.dumps({"config": config, "row_counts": {key: len(value) for key, value in split_rows.items()}, "group_counts": {key: len({row.group_id for row in value}) for key, value in split_rows.items()}}, indent=2), encoding="utf-8")
    with (output / "quarantine.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows({name: item.get(name, "") for name in MANIFEST_COLUMNS} for item in quarantined)
    (output / "dataset_summary.json").write_text(json.dumps({key: summarize(value) for key, value in split_rows.items()}, indent=2), encoding="utf-8")
    freeze_challenge(output / "challenge_freeze.json", split_rows["CHALLENGE"], config)
    if not leakage["passed"]:
        raise RuntimeError("split leakage detected")


if __name__ == "__main__":
    main()
