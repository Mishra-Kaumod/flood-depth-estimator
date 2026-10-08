#!/usr/bin/env python3
"""Report V6 diagnostic stages for one explicitly approved image.

Normal inference uses main.py image. Historical V5 comparator fields are null
with the default primary-only factory; this utility does not execute V5.

This script accepts one path and its expected SHA-256. It does not discover, scan,
or enumerate any dataset, internal test, or external challenge directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.v6_inference import create_v6_pipeline, load_v6_rgb


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="One approved-image V6 diagnostic report")
    parser.add_argument("--image", required=True, help="Explicit development image path")
    parser.add_argument("--expected-sha256", required=True, help="Approval-bound image SHA-256")
    parser.add_argument("--actual-depth-cm", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    image_path = Path(args.image)
    actual_hash = digest(image_path)
    if actual_hash.lower() != args.expected_sha256.lower():
        raise RuntimeError("Image SHA-256 does not match the explicitly approved input")
    image_rgb = load_v6_rgb(image_path)
    result = create_v6_pipeline().predict(image_rgb)
    output = {
        "image_sha256": actual_hash,
        "comparison": result.comparison(args.actual_depth_cm).as_dict(),
        "uncertainty_flags": result.uncertainty.flags,
        "stage_outputs": [
            {
                "stage": stage.stage,
                "numerical_depth_before_cm": stage.numerical_depth_before_cm,
                "numerical_depth_after_cm": stage.numerical_depth_after_cm,
                "numerical_owner": stage.numerical_owner,
                "changed_numerical_depth": stage.changed_numerical_depth,
                "details": dict(stage.details),
            }
            for stage in result.stages
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
