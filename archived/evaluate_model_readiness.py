import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "flood_project.settings")

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402

from flood_api.models import FloodInundationTelemetry  # noqa: E402
from flood_api.services.prediction_policy import harmonize_prediction  # noqa: E402
from flood_api.tasks import process_and_refine_telemetry  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
POSITIVE_MARKERS = ("flood", "inund", "water", "rescue", "storm", "rain", "monsoon")
NEGATIVE_MARKERS = ("dry", "drought", "desert", "no_flood", "noflood")
DEPTH_BANDS = (
    ("0_20", 0.0, 20.0),
    ("20_50", 20.0, 50.0),
    ("50_80", 50.0, 80.0),
    ("80_plus", 80.0, None),
)


def _normalize_path_key(path_value):
    if not path_value:
        return ""
    return str(path_value).replace("\\", "/").lstrip("./")


def _parse_optional_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_optional_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _depth_band_key(depth_cm):
    if depth_cm is None:
        return None
    value = float(depth_cm)
    for band, low, high in DEPTH_BANDS:
        if high is None and value >= low:
            return band
        if high is not None and value >= low and value < high:
            return band
    return None


def infer_label_from_name(name):
    lowered = (name or "").lower()
    if any(token in lowered for token in NEGATIVE_MARKERS):
        return 0
    if any(token in lowered for token in POSITIVE_MARKERS):
        return 1
    return None


def iter_images(root):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_manifest(manifest_path):
    labels = {}
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


def to_repo_relative_key(path_obj, input_dir):
    path_obj = path_obj.resolve()
    input_dir = input_dir.resolve()
    try:
        base_relative = path_obj.relative_to(Path(settings.BASE_DIR))
        return _normalize_path_key(base_relative)
    except ValueError:
        try:
            local_relative = path_obj.relative_to(input_dir)
            return _normalize_path_key(Path(input_dir.name) / local_relative)
        except ValueError:
            return _normalize_path_key(path_obj.name)


def evaluate_dataset(input_dir, limit, manifest_labels):
    input_dir = input_dir.resolve()
    runtime_tmp = Path(getattr(settings, "RUNTIME_TMP_DIR", Path(settings.BASE_DIR) / "tmp_uploads"))
    runtime_tmp.mkdir(parents=True, exist_ok=True)

    candidates = list(iter_images(input_dir))
    if limit:
        candidates = candidates[:limit]

    rows = []
    for idx, source in enumerate(candidates, start=1):
        image_key = to_repo_relative_key(source, input_dir)
        annotation = manifest_labels.get(image_key, {})
        copied = runtime_tmp / f"eval_{idx}_{source.name}"
        shutil.copy2(source, copied)
        start = time.perf_counter()
        out = process_and_refine_telemetry.run(
            image_filepath=str(copied),
            filename=source.name,
            external_context="model_readiness_eval",
            camera_id=f"eval_cam_{idx:03d}",
        )
        elapsed = round(time.perf_counter() - start, 3)
        row = {
            "image": source.name,
            "image_path": image_key,
            "elapsed_sec": elapsed,
            "status": out.get("status"),
        }
        if out.get("status") == "success":
            record = FloodInundationTelemetry.objects.get(id=out["record_id"])
            normalized = harmonize_prediction(
                raw_depth_cm=record.computed_depth_cm,
                water_pct=record.surface_water_confirmed_pct,
                raw_confidence=(record.system_confidence_score_pct or 0.0) / 100.0,
                num_anchors=record.num_reference_objects,
            )
            predicted = 1 if normalized["is_water_confirmed"] else 0
            expected_flood = annotation.get("expected_flood")
            if expected_flood not in (0, 1):
                expected_flood = infer_label_from_name(record.image_name)
            expected_depth = annotation.get("expected_depth_cm")
            scene_type = annotation.get("scene_type")
            if not scene_type:
                scene_type = "unknown"
            contradiction = (
                not normalized["is_water_confirmed"]
                and float(record.surface_water_confirmed_pct or 0.0) >= 35.0
                and float(record.system_confidence_score_pct or 0.0) >= 60.0
            )
            row.update(
                {
                    "record_id": record.id,
                    "depth_cm": float(record.computed_depth_cm or 0.0),
                    "normalized_depth_cm": float(normalized["depth_cm"] or 0.0),
                    "water_pct": float(record.surface_water_confirmed_pct or 0.0),
                    "predicted_flood": predicted,
                    "expected_flood": expected_flood,
                    "expected_depth_cm": expected_depth,
                    "scene_type": scene_type,
                    "label_status": annotation.get("label_status"),
                    "contradiction": contradiction,
                }
            )
        else:
            row["error"] = out.get("message", "unknown error")
        rows.append(row)

    return rows


def build_summary(rows):
    success_rows = [r for r in rows if r.get("status") == "success"]
    latencies = [r["elapsed_sec"] for r in success_rows]
    contradictions = sum(1 for r in success_rows if r.get("contradiction"))
    sorted_latencies = sorted(latencies)
    p95_latency = None
    if sorted_latencies:
        p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
        p95_latency = sorted_latencies[p95_index]

    labeled_flood = [r for r in success_rows if r.get("expected_flood") in (0, 1)]
    tp = sum(1 for r in labeled_flood if r["predicted_flood"] == 1 and r["expected_flood"] == 1)
    tn = sum(1 for r in labeled_flood if r["predicted_flood"] == 0 and r["expected_flood"] == 0)
    fp = sum(1 for r in labeled_flood if r["predicted_flood"] == 1 and r["expected_flood"] == 0)
    fn = sum(1 for r in labeled_flood if r["predicted_flood"] == 0 and r["expected_flood"] == 1)

    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None

    depth_labeled = [r for r in success_rows if r.get("expected_depth_cm") is not None]
    depth_abs_errors = [abs(float(r["normalized_depth_cm"]) - float(r["expected_depth_cm"])) for r in depth_labeled]
    depth_mae = (sum(depth_abs_errors) / len(depth_abs_errors)) if depth_abs_errors else None
    depth_band_samples = {band: [] for band, _, _ in DEPTH_BANDS}
    for row in depth_labeled:
        band = _depth_band_key(row.get("expected_depth_cm"))
        if band is None:
            continue
        depth_band_samples[band].append(abs(float(row["normalized_depth_cm"]) - float(row["expected_depth_cm"])))
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


def evaluate_quality_gates(summary, args):
    failed = []
    classification = summary.get("classification", {})

    if summary.get("labeled_flood_subset_size", 0) < args.min_labeled_flood:
        failed.append(
            f"labeled_flood_subset_size {summary.get('labeled_flood_subset_size', 0)} < {args.min_labeled_flood}"
        )
    if summary.get("labeled_depth_subset_size", 0) < args.min_labeled_depth:
        failed.append(
            f"labeled_depth_subset_size {summary.get('labeled_depth_subset_size', 0)} < {args.min_labeled_depth}"
        )
    if summary.get("barren_subset_size", 0) < args.min_barren:
        failed.append(
            f"barren_subset_size {summary.get('barren_subset_size', 0)} < {args.min_barren}"
        )

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
            failed.append(
                f"depth_band_{band_key}_count {band_count} < {args.min_depth_labels_per_band}"
            )
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
        failed.append(
            f"classification.barren_false_positive_rate {barren_fp_rate} > {args.max_barren_fp_rate}"
        )

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


def main():
    parser = argparse.ArgumentParser(description="Run model readiness evaluation on a local image set.")
    parser.add_argument("--input-dir", default=str(Path(settings.BASE_DIR) / "test_images"), help="Directory containing evaluation images")
    parser.add_argument("--manifest", default=str(Path(settings.BASE_DIR) / "test_images" / "evaluation_manifest.csv"), help="CSV manifest with expected labels/depth")
    parser.add_argument("--limit", type=int, default=0, help="Optional max image count")
    parser.add_argument("--enforce-gates", action="store_true", help="Exit non-zero if quality gates fail")
    parser.add_argument("--min-labeled-flood", type=int, default=50, help="Minimum labeled flood/non-flood samples")
    parser.add_argument("--min-labeled-depth", type=int, default=20, help="Minimum labeled depth samples")
    parser.add_argument("--min-barren", type=int, default=25, help="Minimum barren-scene samples")
    parser.add_argument("--min-f1", type=float, default=0.90, help="Minimum classification F1")
    parser.add_argument("--max-depth-mae", type=float, default=15.0, help="Maximum depth MAE (cm)")
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
        "input_dir": str(input_dir),
        "manifest_path": str(manifest_path),
        "manifest_labels_loaded": len(manifest_labels),
        "summary": summary,
        "quality_gates": gates,
        "rows": rows,
    }

    reports_root = Path(getattr(settings, "RUNTIME_ROOT", Path(settings.BASE_DIR))) / "reports"
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
