#!/usr/bin/env python3
"""Write development-only full traces for explicit approved train/validation rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.segformer_yolo_depthv2_pipeline import get_segformer_yolo_depthv2_pipeline


FORBIDDEN_TERMS = ("internal_test", "external_challenge", "challenge", "labels_test")


def read_rows(manifest: Path, names: set[str]) -> list[dict[str, str]]:
    if any(term in str(manifest).lower() for term in FORBIDDEN_TERMS):
        raise RuntimeError(f"Forbidden manifest for development trace: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if not names or row.get("filename") in names]
    if names and {row.get("filename") for row in selected} != names:
        missing = sorted(names - {row.get("filename") for row in selected})
        raise RuntimeError(f"Requested filenames absent from approved manifest: {missing}")
    return selected


def stage_summary(features: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    keys = [
        "calibration_depth_cm", "pre_output_cap_depth_cm", "output_cap_capped_depth_cm",
        "pre_residual_output_cap_capped_depth_cm", "pre_residual_fusion_depth_cm",
        "residual_fusion_depth_cm", "residual_fusion_applied_depth_cm", "residual_fusion_delta_cm",
        "dynamic_broad_mask_input_depth_cm", "dynamic_broad_mask_estimated_depth_cm",
        "dynamic_broad_mask_resolved_depth_cm", "mask_conditioned_fusion_depth_cm",
        "mask_conditioned_high_flood_corrected_depth_cm", "strong_deep_flood_depth_cm",
        "pre_dry_road_override_depth_cm", "final_aggregation_source", "final_output_reason",
    ]
    return {
        "final_depth_cm": result.get("depth_cm"),
        "confidence": result.get("confidence"),
        "action": result.get("action_trigger"),
        "values": {key: features.get(key) for key in keys if key in features},
        "applied_flags": {
            key: value for key, value in features.items()
            if key.endswith("_applied") and bool(value)
        },
        "statuses": {
            key: value for key, value in features.items()
            if key.endswith("_status") or key.endswith("_reason")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--filenames", required=True, help="Comma-separated explicit filenames")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    names = {name.strip() for name in args.filenames.split(",") if name.strip()}
    rows = read_rows(manifest, names)
    pipeline = get_segformer_yolo_depthv2_pipeline()
    output: list[dict[str, object]] = []
    for row in rows:
        image_path = REPO / row["image_path"]
        # The approved manifest is the authority for row membership. Some clean
        # validation rows retain a historical `images/test` source directory;
        # that name alone is not the consumed internal-test split.
        if not image_path.is_file():
            raise RuntimeError(f"Approved manifest image is unavailable: {image_path}")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        result = pipeline.predict(image)
        features = result.get("structured_features", {}) or {}
        output.append(
            {
                "split_manifest": str(manifest),
                "filename": row["filename"],
                "image_path": row["image_path"],
                "actual_depth_cm": float(row["depth_cm"]),
                "pipeline_trace": result.get("pipeline_trace", []),
                "pipeline_stage_outputs": result.get("pipeline_stage_outputs", []),
                "features": features,
                "stage_summary": stage_summary(features, result),
            }
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"wrote={output_path} traces={len(output)}", flush=True)


if __name__ == "__main__":
    main()
