"""Training-readiness data contract for future V6 development data.

This module validates metadata and split integrity only. It never loads inference
models and never converts missing depth labels into numeric zero.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


MANIFEST_COLUMNS = (
    "image_id", "filename", "sha256", "scene_type", "depth_cm", "label_confidence",
    "measurement_source", "source_session_id", "group_id", "physical_reference",
    "depth_eligible", "classification_only", "quarantine_reason", "visual_near_duplicate_status",
)


class SceneType(str, Enum):
    DRY = "DRY"
    WET_NO_FLOOD = "WET_NO_FLOOD"
    FLOOD_0_10 = "FLOOD_0_10"
    FLOOD_10_20 = "FLOOD_10_20"
    FLOOD_20_50 = "FLOOD_20_50"
    FLOOD_50_75 = "FLOOD_50_75"
    FLOOD_75_PLUS = "FLOOD_75_PLUS"


FLOOD_SCENES = frozenset({
    SceneType.FLOOD_0_10, SceneType.FLOOD_10_20, SceneType.FLOOD_20_50,
    SceneType.FLOOD_50_75, SceneType.FLOOD_75_PLUS,
})
CONFIDENCES = frozenset({"HIGH", "MEDIUM", "LOW"})
MEASUREMENT_SOURCES = frozenset({"instrumented", "measured_reference", "expert_visual_estimate", "video_annotation", "unknown"})
VISUAL_STATUS = frozenset({"not_run", "no_match", "near_duplicate", "visual_duplicate", "needs_review"})


def parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field_name} must be true or false")


def parse_optional_depth(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    depth = float(value)
    if depth != depth or depth == float("inf") or depth == float("-inf"):
        raise ValueError("depth_cm must be finite")
    return depth


def depth_bucket(depth_cm: Optional[float]) -> str:
    if depth_cm is None:
        return "not_depth_eligible"
    if depth_cm <= 10:
        return "0-10"
    if depth_cm <= 20:
        return "10-20"
    if depth_cm <= 50:
        return "20-50"
    if depth_cm <= 75:
        return "50-75"
    return "75+"


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    filename: str
    sha256: Optional[str]
    scene_type: SceneType
    depth_cm: Optional[float]
    label_confidence: str
    measurement_source: str
    source_session_id: str
    group_id: str
    physical_reference: Optional[str]
    depth_eligible: bool
    classification_only: bool
    quarantine_reason: Optional[str] = None
    visual_near_duplicate_status: str = "not_run"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ManifestRow":
        missing = [name for name in ("image_id", "filename", "scene_type", "label_confidence", "measurement_source", "source_session_id", "group_id", "depth_eligible", "classification_only") if not str(raw.get(name, "")).strip()]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        try:
            scene = SceneType(str(raw["scene_type"]).strip())
        except ValueError as exc:
            raise ValueError(f"invalid scene_type: {raw.get('scene_type')}") from exc
        confidence = str(raw["label_confidence"]).strip().upper()
        source = str(raw["measurement_source"]).strip()
        visual = str(raw.get("visual_near_duplicate_status") or "not_run").strip()
        if confidence not in CONFIDENCES:
            raise ValueError(f"invalid label_confidence: {confidence}")
        if source not in MEASUREMENT_SOURCES:
            raise ValueError(f"invalid measurement_source: {source}")
        if visual not in VISUAL_STATUS:
            raise ValueError(f"invalid visual_near_duplicate_status: {visual}")
        row = cls(
            image_id=str(raw["image_id"]).strip(), filename=str(raw["filename"]).strip(),
            sha256=(str(raw.get("sha256")).strip().lower() or None), scene_type=scene,
            depth_cm=parse_optional_depth(raw.get("depth_cm")), label_confidence=confidence,
            measurement_source=source, source_session_id=str(raw["source_session_id"]).strip(),
            group_id=str(raw["group_id"]).strip(), physical_reference=(str(raw.get("physical_reference")).strip() or None),
            depth_eligible=parse_bool(raw["depth_eligible"], "depth_eligible"),
            classification_only=parse_bool(raw["classification_only"], "classification_only"),
            quarantine_reason=(str(raw.get("quarantine_reason")).strip() or None), visual_near_duplicate_status=visual,
        )
        row.validate()
        return row

    def validate(self) -> None:
        if self.scene_type in {SceneType.DRY, SceneType.WET_NO_FLOOD}:
            if self.depth_cm is not None:
                raise ValueError("DRY/WET_NO_FLOOD depth_cm must be blank; do not encode as zero")
            if self.depth_eligible or not self.classification_only:
                raise ValueError("DRY/WET_NO_FLOOD must be classification_only=true and depth_eligible=false")
        else:
            if self.depth_cm is None or self.depth_cm <= 0:
                raise ValueError("flood scene requires positive depth_cm; zero/flood conflict is quarantined")
            if not self.depth_eligible or self.classification_only:
                raise ValueError("flood scene must be depth_eligible=true and classification_only=false")

    def to_mapping(self) -> dict[str, str]:
        data = asdict(self)
        data["scene_type"] = self.scene_type.value
        data["depth_cm"] = "" if self.depth_cm is None else str(self.depth_cm)
        data["sha256"] = self.sha256 or ""
        data["physical_reference"] = self.physical_reference or ""
        data["quarantine_reason"] = self.quarantine_reason or ""
        data["depth_eligible"] = str(self.depth_eligible).lower()
        data["classification_only"] = str(self.classification_only).lower()
        return {name: str(data[name]) for name in MANIFEST_COLUMNS}


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"manifest header must equal: {', '.join(MANIFEST_COLUMNS)}")
        return [ManifestRow.from_mapping(raw) for raw in reader]


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(row.to_mapping() for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_hash(path: Path) -> str:
    return sha256_file(path)


def quarantine_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[ManifestRow], list[dict[str, Any]]]:
    valid: list[ManifestRow] = []
    quarantined: list[dict[str, Any]] = []
    for raw in rows:
        try:
            valid.append(ManifestRow.from_mapping(raw))
        except (TypeError, ValueError) as exc:
            item = dict(raw)
            item["quarantine_reason"] = str(exc)
            quarantined.append(item)
    return valid, quarantined


def freeze_challenge(path: Path, rows: Sequence[ManifestRow], split_config: Mapping[str, Any]) -> None:
    payload = {
        "split": "CHALLENGE",
        "frozen_on": date.today().isoformat(),
        "split_config": dict(split_config),
        "image_ids": [row.image_id for row in rows],
        "group_ids": sorted({row.group_id for row in rows}),
        "sha256": [row.sha256 for row in rows],
        "manifest_hash": manifest_hash(path.with_name("challenge_manifest.csv")),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def assert_manifest_allowed_for_use(manifest_path: Path, purpose: str, final_evaluation: bool = False, candidate_freeze: Optional[Path] = None) -> None:
    """Default-deny Challenge rows for future training, tuning, and development evaluation."""
    is_challenge = manifest_path.name.lower() == "challenge_manifest.csv"
    if not is_challenge:
        return
    if purpose in {"training", "tuning"}:
        raise RuntimeError("CHALLENGE manifest is forbidden for training and tuning")
    if purpose == "evaluation" and (not final_evaluation or candidate_freeze is None or not candidate_freeze.is_file()):
        raise RuntimeError("CHALLENGE evaluation requires final_evaluation mode and an existing frozen candidate artifact")
