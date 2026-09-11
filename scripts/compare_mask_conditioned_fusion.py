import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.segformer_yolo_depthv2_pipeline import get_segformer_yolo_depthv2_pipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = normalize_path(row.get("image_path"))
            expected_depth = parse_float(row.get("expected_depth_cm"))
            if not image_path or expected_depth is None:
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "actual_depth_cm": expected_depth,
                    "expected_flood": row.get("expected_flood", ""),
                    "scene_type": row.get("scene_type", ""),
                    "label_status": row.get("label_status", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def resolve_image_path(repo_root: Path, input_dir: Path, manifest_image_path: str) -> Path:
    raw = Path(manifest_image_path)
    candidates = [
        repo_root / raw,
        input_dir / raw,
        input_dir / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    raise FileNotFoundError(manifest_image_path)


def closer_model(
    actual_depth: float,
    current_depth: Optional[float],
    mask_conditioned_depth: Optional[float],
) -> str:
    if current_depth is None and mask_conditioned_depth is None:
        return "missing_both"
    if current_depth is None:
        return "mask_conditioned_fusion"
    if mask_conditioned_depth is None:
        return "current_final"

    current_error = abs(current_depth - actual_depth)
    mask_error = abs(mask_conditioned_depth - actual_depth)
    if math.isclose(current_error, mask_error, abs_tol=0.5):
        return "tie"
    return "current_final" if current_error < mask_error else "mask_conditioned_fusion"


def run_comparison(input_dir: Path, manifest: Path, limit: int) -> List[Dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = get_segformer_yolo_depthv2_pipeline()
    manifest_rows = load_manifest(manifest)
    if limit:
        manifest_rows = manifest_rows[:limit]

    results: List[Dict[str, Any]] = []
    for item in manifest_rows:
        start = time.perf_counter()
        result: Dict[str, Any] = {
            "image": Path(item["image_path"]).name,
            "image_path": item["image_path"],
            "actual_depth_cm": item["actual_depth_cm"],
            "scene_type": item["scene_type"],
            "label_status": item["label_status"],
            "expected_flood": item["expected_flood"],
            "notes": item["notes"],
        }

        try:
            image_path = resolve_image_path(repo_root, input_dir, item["image_path"])
            image_rgb = np.array(Image.open(image_path).convert("RGB"))
            prediction = pipeline.predict(image_rgb)

            features = prediction.get("structured_features", {})
            if not isinstance(features, dict):
                features = {}

            current_depth = parse_float(prediction.get("depth_cm"))
            mask_conditioned_depth = parse_float(
                prediction.get("mask_conditioned_fusion_depth_cm")
                or features.get("mask_conditioned_fusion_depth_cm")
            )
            actual_depth = float(item["actual_depth_cm"])

            current_error = abs(current_depth - actual_depth) if current_depth is not None else None
            mask_error = (
                abs(mask_conditioned_depth - actual_depth)
                if mask_conditioned_depth is not None
                else None
            )

            result.update(
                {
                    "status": "success",
                    "current_final_depth_cm": current_depth,
                    "mask_conditioned_fusion_depth_cm": mask_conditioned_depth,
                    "current_final_error_cm": round(current_error, 3) if current_error is not None else "",
                    "mask_conditioned_fusion_error_cm": round(mask_error, 3) if mask_error is not None else "",
                    "which_one_is_closer": closer_model(actual_depth, current_depth, mask_conditioned_depth),
                    "final_severity": prediction.get("severity", {}).get("level")
                    if isinstance(prediction.get("severity"), dict)
                    else "",
                    "water_coverage_pct": round(float(prediction.get("water_coverage", 0.0)) * 100.0, 3),
                    "reference_count": prediction.get("reference_count", ""),
                    "elapsed_sec": round(time.perf_counter() - start, 3),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "current_final_depth_cm": "",
                    "mask_conditioned_fusion_depth_cm": "",
                    "which_one_is_closer": "error",
                    "elapsed_sec": round(time.perf_counter() - start, 3),
                }
            )
        results.append(result)

    return results


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    success_rows = [row for row in rows if row.get("status") == "success"]
    current_errors = [
        float(row["current_final_error_cm"])
        for row in success_rows
        if row.get("current_final_error_cm") != ""
    ]
    mask_errors = [
        float(row["mask_conditioned_fusion_error_cm"])
        for row in success_rows
        if row.get("mask_conditioned_fusion_error_cm") != ""
    ]
    winners: Dict[str, int] = {}
    for row in success_rows:
        winner = str(row.get("which_one_is_closer", "unknown"))
        winners[winner] = winners.get(winner, 0) + 1

    return {
        "rows": len(rows),
        "success": len(success_rows),
        "failed": len(rows) - len(success_rows),
        "current_final_mae_cm": round(sum(current_errors) / len(current_errors), 3)
        if current_errors
        else None,
        "mask_conditioned_fusion_mae_cm": round(sum(mask_errors) / len(mask_errors), 3)
        if mask_errors
        else None,
        "winner_counts": winners,
    }


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "image_path",
        "actual_depth_cm",
        "current_final_depth_cm",
        "mask_conditioned_fusion_depth_cm",
        "current_final_error_cm",
        "mask_conditioned_fusion_error_cm",
        "which_one_is_closer",
        "scene_type",
        "label_status",
        "expected_flood",
        "final_severity",
        "water_coverage_pct",
        "reference_count",
        "elapsed_sec",
        "status",
        "error",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current final depth against MaskConditionedFusion depth."
    )
    parser.add_argument("--input-dir", default="evaluation_data/images")
    parser.add_argument("--manifest", default="evaluation_data/evaluation_manifest.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    output_path = (
        Path(args.output)
        if args.output
        else Path("reports")
        / f"mask_conditioned_fusion_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    rows = run_comparison(
        input_dir=Path(args.input_dir),
        manifest=Path(args.manifest),
        limit=args.limit,
    )
    write_csv(rows, output_path)
    summary = summarize(rows)

    print(f"Report: {output_path}")
    print(f"Images processed: {summary['rows']}")
    print(f"Successful: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    print(f"Current final MAE: {summary['current_final_mae_cm']} cm")
    print(f"MaskConditionedFusion MAE: {summary['mask_conditioned_fusion_mae_cm']} cm")
    print(f"Closer counts: {summary['winner_counts']}")


if __name__ == "__main__":
    main()



