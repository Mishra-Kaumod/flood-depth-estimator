import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from src.segformer_yolo_depthv2_pipeline import get_segformer_yolo_depthv2_pipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
POSITIVE_MARKERS = ("flood", "inund", "water", "rescue", "storm", "rain", "monsoon")
NEGATIVE_MARKERS = ("dry", "drought", "desert", "no_flood", "noflood", "barren")
DEPTH_BANDS = (
    ("0_20", 0.0, 20.0),
    ("20_50", 20.0, 50.0),
    ("50_80", 50.0, 80.0),
    ("80_plus", 80.0, None),
)


def _normalize_path_key(path_value: Any) -> str:
    if not path_value:
        return ""
    return str(path_value).replace("\\", "/").lstrip("./")


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _depth_band_key(depth_cm: Optional[float]) -> Optional[str]:
    if depth_cm is None:
        return None
    value = float(depth_cm)
    for band, low, high in DEPTH_BANDS:
        if high is None and value >= low:
            return band
        if high is not None and low <= value < high:
            return band
    return None


def infer_label_from_name(name: str) -> Optional[int]:
    lowered = (name or "").lower()
    if any(token in lowered for token in NEGATIVE_MARKERS):
        return 0
    if any(token in lowered for token in POSITIVE_MARKERS):
        return 1
    return None


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    labels: Dict[str, Dict[str, Any]] = {}
    if manifest_path is None or not manifest_path.exists():
        return labels

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _normalize_path_key(row.get("image_path"))
            if not key:
                continue
            labels[key] = {
                "expected_flood": _parse_optional_int(row.get("expected_flood")),
                "expected_depth_cm": _parse_optional_float(row.get("expected_depth_cm")),
                "scene_type": (row.get("scene_type") or "").strip().lower() or None,
                "label_status": (row.get("label_status") or "").strip().lower() or None,
                "notes": (row.get("notes") or "").strip(),
            }
    return labels


def resolve_manifest_row(
    image_path: Path,
    repo_root: Path,
    input_dir: Path,
    manifest_labels: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    candidates: List[str] = []

    try:
        repo_relative = image_path.resolve().relative_to(repo_root.resolve())
        candidates.append(_normalize_path_key(repo_relative))
    except ValueError:
        pass

    try:
        input_relative = image_path.resolve().relative_to(input_dir.resolve())
        candidates.append(_normalize_path_key(Path(input_dir.name) / input_relative))
        candidates.append(_normalize_path_key(input_relative))
    except ValueError:
        pass

    candidates.append(_normalize_path_key(image_path.name))

    for key in candidates:
        if key in manifest_labels:
            result = dict(manifest_labels[key])
            result["image_path"] = key
            return result

    fallback_key = candidates[0] if candidates else _normalize_path_key(image_path.name)
    return {
        "image_path": fallback_key,
        "expected_flood": None,
        "expected_depth_cm": None,
        "scene_type": None,
        "label_status": None,
        "notes": "",
    }


def predict_flood_label(
    eval_depth_cm: Optional[float],
    water_coverage_pct: float,
    water_confidence: float,
    water_detection_unreliable: bool,
) -> int:
    if water_detection_unreliable:
        return 0
    if eval_depth_cm is not None and eval_depth_cm >= 5.0:
        return 1
    if water_coverage_pct >= 8.0 and water_confidence >= 0.45:
        return 1
    return 0


def detect_contradiction(
    predicted_flood: int,
    eval_depth_cm: Optional[float],
    water_coverage_pct: float,
    water_confidence: float,
    water_detection_unreliable: bool,
) -> bool:
    if predicted_flood == 0 and eval_depth_cm is not None and eval_depth_cm >= 20.0:
        return True
    if predicted_flood == 0 and water_coverage_pct >= 35.0 and water_confidence >= 0.60:
        return True
    if predicted_flood == 1 and water_detection_unreliable:
        return True
    if predicted_flood == 1 and eval_depth_cm is not None and eval_depth_cm < 2.0 and water_coverage_pct < 2.0:
        return True
    return False


def evaluate_dataset(
    input_dir: Path,
    limit: int,
    manifest_labels: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    repo_root = Path(__file__).resolve().parent
    pipeline = get_segformer_yolo_depthv2_pipeline()

    candidates = list(iter_images(input_dir))
    if limit:
        candidates = candidates[:limit]

    rows: List[Dict[str, Any]] = []
    for source in candidates:
        annotation = resolve_manifest_row(
            image_path=source,
            repo_root=repo_root,
            input_dir=input_dir,
            manifest_labels=manifest_labels,
        )
        expected_flood = annotation.get("expected_flood")
        if expected_flood not in (0, 1):
            expected_flood = infer_label_from_name(source.name)
        expected_depth = annotation.get("expected_depth_cm")
        scene_type = annotation.get("scene_type") or ("barren" if expected_flood == 0 else "unknown")

        start = time.perf_counter()
        row: Dict[str, Any] = {
            "image": source.name,
            "image_path": annotation.get("image_path"),
        }
        try:
            image_rgb = np.array(Image.open(source).convert("RGB"))
            out = pipeline.predict(image_rgb)
            elapsed = round(time.perf_counter() - start, 3)

            depth_cm = _parse_optional_float(out.get("depth_cm"))
            provisional_depth_cm = _parse_optional_float(out.get("provisional_depth_cm"))
            eval_depth_cm = depth_cm if depth_cm is not None else provisional_depth_cm
            water_coverage_raw = _parse_optional_float(out.get("water_coverage")) or 0.0
            water_coverage_pct = water_coverage_raw * 100.0 if water_coverage_raw <= 1.0 else water_coverage_raw
            water_confidence = _parse_optional_float(out.get("water_confidence")) or 0.0
            water_detection_unreliable = bool(out.get("water_detection_unreliable"))
            no_reference_warning = bool(out.get("no_reference_warning"))
            predicted_flood = predict_flood_label(
                eval_depth_cm=eval_depth_cm,
                water_coverage_pct=water_coverage_pct,
                water_confidence=water_confidence,
                water_detection_unreliable=water_detection_unreliable,
            )
            contradiction = detect_contradiction(
                predicted_flood=predicted_flood,
                eval_depth_cm=eval_depth_cm,
                water_coverage_pct=water_coverage_pct,
                water_confidence=water_confidence,
                water_detection_unreliable=water_detection_unreliable,
            )

            severity = out.get("severity")
            severity_level = severity.get("level") if isinstance(severity, dict) else None

            row.update(
                {
                    "status": "success",
                    "elapsed_sec": elapsed,
                    "method": out.get("method"),
                    "predicted_flood": predicted_flood,
                    "expected_flood": expected_flood,
                    "depth_cm": depth_cm,
                    "provisional_depth_cm": provisional_depth_cm,
                    "eval_depth_cm": eval_depth_cm,
                    "expected_depth_cm": expected_depth,
                    "severity_level": severity_level,
                    "confidence": _parse_optional_float(out.get("confidence")),
                    "water_confidence": round(water_confidence, 4),
                    "water_coverage_pct": round(float(water_coverage_pct), 4),
                    "scene_type": scene_type,
                    "label_status": annotation.get("label_status"),
                    "notes": annotation.get("notes"),
                    "no_reference_warning": no_reference_warning,
                    "water_detection_unreliable": water_detection_unreliable,
                    "contradiction": contradiction,
                }
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - start, 3)
            row.update(
                {
                    "status": "error",
                    "elapsed_sec": elapsed,
                    "error": str(exc),
                    "expected_flood": expected_flood,
                    "expected_depth_cm": expected_depth,
                    "scene_type": scene_type,
                }
            )
        rows.append(row)

    return rows


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    success_rows = [r for r in rows if r.get("status") == "success"]
    latencies = [float(r.get("elapsed_sec", 0.0)) for r in success_rows]
    sorted_latencies = sorted(latencies)
    p95_latency = None
    if sorted_latencies:
        p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
        p95_latency = sorted_latencies[p95_index]

    labeled_flood = [r for r in success_rows if r.get("expected_flood") in (0, 1)]
    tp = sum(1 for r in labeled_flood if r.get("predicted_flood") == 1 and r.get("expected_flood") == 1)
    tn = sum(1 for r in labeled_flood if r.get("predicted_flood") == 0 and r.get("expected_flood") == 0)
    fp = sum(1 for r in labeled_flood if r.get("predicted_flood") == 1 and r.get("expected_flood") == 0)
    fn = sum(1 for r in labeled_flood if r.get("predicted_flood") == 0 and r.get("expected_flood") == 1)

    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and (precision + recall)
        else None
    )

    depth_labeled = [
        r
        for r in success_rows
        if r.get("expected_depth_cm") is not None and r.get("eval_depth_cm") is not None
    ]
    depth_abs_errors = [
        abs(float(r["eval_depth_cm"]) - float(r["expected_depth_cm"]))
        for r in depth_labeled
    ]
    depth_squared_errors = [
        (float(r["eval_depth_cm"]) - float(r["expected_depth_cm"])) ** 2
        for r in depth_labeled
    ]
    depth_mae = (sum(depth_abs_errors) / len(depth_abs_errors)) if depth_abs_errors else None
    depth_rmse = math.sqrt(sum(depth_squared_errors) / len(depth_squared_errors)) if depth_squared_errors else None
    depth_r2 = None
    if depth_labeled:
        expected_values = [float(r["expected_depth_cm"]) for r in depth_labeled]
        predicted_values = [float(r["eval_depth_cm"]) for r in depth_labeled]
        expected_mean = sum(expected_values) / len(expected_values)
        total_sum_squares = sum((value - expected_mean) ** 2 for value in expected_values)
        residual_sum_squares = sum((actual - predicted) ** 2 for actual, predicted in zip(expected_values, predicted_values))
        if total_sum_squares > 0:
            depth_r2 = 1.0 - (residual_sum_squares / total_sum_squares)

    depth_band_samples: Dict[str, List[float]] = {band: [] for band, _, _ in DEPTH_BANDS}
    for row in depth_labeled:
        band = _depth_band_key(_parse_optional_float(row.get("expected_depth_cm")))
        if band is None:
            continue
        error = abs(float(row["eval_depth_cm"]) - float(row["expected_depth_cm"]))
        depth_band_samples[band].append(error)

    depth_mae_by_band = {
        band: {
            "count": len(errors),
            "mae_cm": round(sum(errors) / len(errors), 3) if errors else None,
        }
        for band, errors in depth_band_samples.items()
    }

    barren_rows = [r for r in success_rows if (r.get("scene_type") or "").lower() == "barren"]
    barren_fp_count = sum(1 for r in barren_rows if r.get("predicted_flood") == 1)
    barren_fp_rate = (barren_fp_count / len(barren_rows)) if barren_rows else None

    contradictions = sum(1 for r in success_rows if bool(r.get("contradiction")))

    return {
        "images_processed": len(rows),
        "images_success": len(success_rows),
        "images_failed": len(rows) - len(success_rows),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "max_latency_sec": max(latencies) if latencies else None,
        "p95_latency_sec": p95_latency,
        "contradiction_count": contradictions,
        "labeled_flood_subset_size": len(labeled_flood),
        "labeled_depth_subset_size": len(depth_labeled),
        "barren_subset_size": len(barren_rows),
        "depth_mae_cm": round(depth_mae, 3) if depth_mae is not None else None,
        "depth_rmse_cm": round(depth_rmse, 3) if depth_rmse is not None else None,
        "depth_r2": round(depth_r2, 4) if depth_r2 is not None else None,
        "depth_mae_by_band": depth_mae_by_band,
        "classification": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "barren_false_positive_count": barren_fp_count,
            "barren_false_positive_rate": round(barren_fp_rate, 4) if barren_fp_rate is not None else None,
        },
    }


def evaluate_quality_gates(summary: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    failed: List[str] = []
    classification = summary.get("classification", {})

    if summary.get("labeled_flood_subset_size", 0) < args.min_labeled_flood:
        failed.append(f"labeled_flood_subset_size {summary.get('labeled_flood_subset_size', 0)} < {args.min_labeled_flood}")
    if summary.get("labeled_depth_subset_size", 0) < args.min_labeled_depth:
        failed.append(f"labeled_depth_subset_size {summary.get('labeled_depth_subset_size', 0)} < {args.min_labeled_depth}")
    if summary.get("barren_subset_size", 0) < args.min_barren:
        failed.append(f"barren_subset_size {summary.get('barren_subset_size', 0)} < {args.min_barren}")

    f1 = classification.get("f1")
    if f1 is None:
        failed.append("classification.f1 is missing")
    elif f1 < args.min_f1:
        failed.append(f"classification.f1 {f1} < {args.min_f1}")

    depth_mae = summary.get("depth_mae_cm")
    if depth_mae is None:
        failed.append("depth_mae_cm is missing")
    elif depth_mae > args.max_depth_mae:
        failed.append(f"depth_mae_cm {depth_mae} > {args.max_depth_mae}")

    if args.min_depth_r2 is not None:
        depth_r2 = summary.get("depth_r2")
        if depth_r2 is None:
            failed.append("depth_r2 is missing")
        elif depth_r2 < args.min_depth_r2:
            failed.append(f"depth_r2 {depth_r2} < {args.min_depth_r2}")

    band_thresholds = {
        "0_20": args.max_mae_0_20,
        "20_50": args.max_mae_20_50,
        "50_80": args.max_mae_50_80,
        "80_plus": args.max_mae_80_plus,
    }
    depth_bands = summary.get("depth_mae_by_band", {})
    for band_key, threshold in band_thresholds.items():
        band_data = depth_bands.get(band_key, {})
        band_count = int(band_data.get("count") or 0)
        band_mae = band_data.get("mae_cm")
        if band_count < args.min_depth_labels_per_band:
            failed.append(f"depth_band_{band_key}_count {band_count} < {args.min_depth_labels_per_band}")
            continue
        if band_mae is None:
            failed.append(f"depth_band_{band_key}_mae missing")
            continue
        if band_mae > threshold:
            failed.append(f"depth_band_{band_key}_mae {band_mae} > {threshold}")

    barren_fp_rate = classification.get("barren_false_positive_rate")
    if barren_fp_rate is None:
        failed.append("classification.barren_false_positive_rate is missing")
    elif barren_fp_rate > args.max_barren_fp_rate:
        failed.append(f"classification.barren_false_positive_rate {barren_fp_rate} > {args.max_barren_fp_rate}")

    p95_latency = summary.get("p95_latency_sec")
    if p95_latency is None:
        failed.append("p95_latency_sec is missing")
    elif p95_latency > args.max_p95_latency:
        failed.append(f"p95_latency_sec {p95_latency} > {args.max_p95_latency}")

    contradictions = summary.get("contradiction_count")
    if contradictions is None:
        failed.append("contradiction_count is missing")
    elif contradictions > args.max_contradictions:
        failed.append(f"contradiction_count {contradictions} > {args.max_contradictions}")

    return {
        "passed": len(failed) == 0,
        "failed_gates": failed,
        "thresholds": {
            "min_labeled_flood": args.min_labeled_flood,
            "min_labeled_depth": args.min_labeled_depth,
            "min_barren": args.min_barren,
            "min_f1": args.min_f1,
            "max_depth_mae": args.max_depth_mae,
            "min_depth_r2": args.min_depth_r2,
            "min_depth_labels_per_band": args.min_depth_labels_per_band,
            "max_mae_0_20": args.max_mae_0_20,
            "max_mae_20_50": args.max_mae_20_50,
            "max_mae_50_80": args.max_mae_50_80,
            "max_mae_80_plus": args.max_mae_80_plus,
            "max_barren_fp_rate": args.max_barren_fp_rate,
            "max_p95_latency": args.max_p95_latency,
            "max_contradictions": args.max_contradictions,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model readiness evaluation on local test images.")
    parser.add_argument("--input-dir", default="test_images", help="Directory containing evaluation images")
    parser.add_argument("--manifest", default="test_images/evaluation_manifest_labeled.csv", help="CSV manifest with expected labels/depth")
    parser.add_argument("--limit", type=int, default=0, help="Optional max image count")
    parser.add_argument("--enforce-gates", action="store_true", help="Exit non-zero if quality gates fail")
    parser.add_argument("--min-labeled-flood", type=int, default=50, help="Minimum labeled flood/non-flood samples")
    parser.add_argument("--min-labeled-depth", type=int, default=20, help="Minimum labeled depth samples")
    parser.add_argument("--min-barren", type=int, default=25, help="Minimum barren-scene samples")
    parser.add_argument("--min-f1", type=float, default=0.90, help="Minimum classification F1")
    parser.add_argument("--max-depth-mae", type=float, default=15.0, help="Maximum depth MAE (cm)")
    parser.add_argument("--min-depth-r2", type=float, default=None, help="Optional minimum R2 score for depth regression")
    parser.add_argument("--min-depth-labels-per-band", type=int, default=3, help="Minimum labeled depth samples required in each depth band")
    parser.add_argument("--max-mae-0-20", type=float, default=8.0, help="Maximum MAE for 0-20cm depth band")
    parser.add_argument("--max-mae-20-50", type=float, default=12.0, help="Maximum MAE for 20-50cm depth band")
    parser.add_argument("--max-mae-50-80", type=float, default=15.0, help="Maximum MAE for 50-80cm depth band")
    parser.add_argument("--max-mae-80-plus", type=float, default=20.0, help="Maximum MAE for 80cm+ depth band")
    parser.add_argument("--max-barren-fp-rate", type=float, default=0.05, help="Maximum barren false positive rate")
    parser.add_argument("--max-p95-latency", type=float, default=8.0, help="Maximum p95 latency seconds")
    parser.add_argument("--max-contradictions", type=int, default=0, help="Maximum contradictory outputs allowed")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    manifest_path = Path(args.manifest)
    manifest_labels = load_manifest(manifest_path)

    rows = evaluate_dataset(input_dir=input_dir, limit=args.limit, manifest_labels=manifest_labels)
    summary = build_summary(rows)
    gates = evaluate_quality_gates(summary, args)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_labels_loaded": len(manifest_labels),
        "summary": summary,
        "quality_gates": gates,
        "rows": rows,
    }

    reports_root = Path("reports")
    reports_root.mkdir(parents=True, exist_ok=True)
    output_path = reports_root / f"model_readiness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Report: {output_path}")
    print(json.dumps(summary, indent=2))
    print(json.dumps({"quality_gates": gates}, indent=2))

    if args.enforce_gates and not gates["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
