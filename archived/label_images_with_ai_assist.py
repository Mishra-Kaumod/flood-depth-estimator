import argparse
import csv
import os
import time
from pathlib import Path

import cv2

from core_logic import TripleEnginePipeline, estimate_flood_depth
from flood_api.services.prediction_policy import harmonize_prediction


def _parse_optional_int(value):
    text = ("" if value is None else str(value)).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_optional_float(value):
    text = ("" if value is None else str(value)).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_path(path_text):
    return str(path_text).replace("\\", "/").lstrip("./")


def _infer_scene_type(predicted_flood, water_pct, normalized_depth_cm):
    if predicted_flood == 1:
        return "flood"
    if normalized_depth_cm == 0.0 and water_pct <= 3.0:
        return "barren"
    return "non_flood"


def _upsert_note(existing, message):
    existing = (existing or "").strip()
    if existing == "":
        return message
    if message in existing:
        return existing
    return f"{existing} | {message}"


def process_manifest(manifest_path, input_root, output_path, limit, min_confidence, fill_expected):
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        input_fields = list(reader.fieldnames or [])

    ai_fields = [
        "ai_suggested_flood",
        "ai_suggested_depth_cm",
        "ai_suggested_scene_type",
        "ai_confidence",
        "ai_status",
        "ai_latency_sec",
        "ai_model_version",
        "ai_notes",
    ]
    output_fields = input_fields + [f for f in ai_fields if f not in input_fields]

    runtime_root = Path(os.getenv("FLOOD_RUNTIME_ROOT", r"E:\flood_runtime"))
    if not runtime_root.exists():
        runtime_root = Path.cwd()
    runtime_tmp = runtime_root / "tmp_uploads_ai_assist"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    ml_pipeline = TripleEnginePipeline()

    pending_indices = []
    for idx, row in enumerate(rows):
        expected = _parse_optional_int(row.get("expected_flood"))
        if expected in (0, 1):
            continue
        pending_indices.append(idx)

    if limit and limit > 0:
        pending_indices = pending_indices[:limit]

    total = len(pending_indices)
    done = 0
    applied = 0
    failed = 0

    for pos, row_idx in enumerate(pending_indices, start=1):
        row = rows[row_idx]
        image_rel = _normalize_path(row.get("image_path", ""))
        image_abs = (input_root / image_rel).resolve()
        if not image_abs.exists():
            row["ai_status"] = "file_missing"
            row["ai_notes"] = _upsert_note(row.get("ai_notes"), "Image file not found for AI labeling.")
            failed += 1
            continue

        image_matrix = cv2.imread(str(image_abs))
        if image_matrix is None:
            row["ai_status"] = "corrupt_image"
            row["ai_notes"] = _upsert_note(row.get("ai_notes"), "Image could not be decoded.")
            failed += 1
            continue

        start = time.perf_counter()
        flood_prob = ml_pipeline.predict_flood_probability(image_matrix)
        depth_result = estimate_flood_depth(image_matrix)
        elapsed = round(time.perf_counter() - start, 3)

        if depth_result.get("status") != "success":
            row["ai_status"] = "inference_error"
            row["ai_latency_sec"] = str(elapsed)
            row["ai_notes"] = _upsert_note(row.get("ai_notes"), depth_result.get("message", "Inference failed"))
            failed += 1
            continue

        raw_depth_cm = float(depth_result.get("estimated_depth_cm", 0.0))
        ensemble_confidence = float(depth_result.get("ensemble_confidence", 0.0))
        raw_confidence = max(float(flood_prob), ensemble_confidence)
        strategy = depth_result.get("calculation_mode", "")
        num_anchors = 0 if "No anchor" in strategy else 1
        water_pct = float(flood_prob * 100.0)

        normalized = harmonize_prediction(
            raw_depth_cm=raw_depth_cm,
            water_pct=water_pct,
            raw_confidence=raw_confidence,
            num_anchors=num_anchors,
        )

        predicted_flood = 1 if normalized["is_water_confirmed"] else 0
        suggested_depth = float(normalized["depth_cm"] or 0.0)
        confidence = max(raw_confidence, water_pct / 100.0)
        confidence = max(0.0, min(1.0, confidence))
        scene_type = _infer_scene_type(predicted_flood, water_pct, suggested_depth)

        row["ai_suggested_flood"] = str(predicted_flood)
        row["ai_suggested_depth_cm"] = f"{suggested_depth:.1f}"
        row["ai_suggested_scene_type"] = scene_type
        row["ai_confidence"] = f"{confidence:.3f}"
        row["ai_latency_sec"] = str(elapsed)
        row["ai_model_version"] = "pipeline+policy-v1"
        row["ai_status"] = "high_confidence" if confidence >= min_confidence else "low_confidence"
        row["ai_notes"] = _upsert_note(
            row.get("ai_notes"),
            "AI-assisted suggestion by Copilot pipeline. Human review required before production use.",
        )

        if fill_expected and confidence >= min_confidence:
            if _parse_optional_int(row.get("expected_flood")) not in (0, 1):
                row["expected_flood"] = str(predicted_flood)
            if _parse_optional_float(row.get("expected_depth_cm")) is None and predicted_flood == 1:
                row["expected_depth_cm"] = f"{suggested_depth:.1f}"
            current_scene = (row.get("scene_type") or "").strip().lower()
            if current_scene in ("", "unknown"):
                row["scene_type"] = scene_type
            row["label_status"] = "ai_suggested"
            row["notes"] = _upsert_note(
                row.get("notes"),
                f"AI-assisted seed (confidence={confidence:.3f}); requires human verification.",
            )
            applied += 1

        done += 1
        if pos % 25 == 0 or pos == total:
            print(f"Processed {pos}/{total} images...")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Output manifest: {output_path}")
    print(f"Rows considered for AI assist: {total}")
    print(f"Rows processed successfully: {done}")
    print(f"Rows failed: {failed}")
    print(f"Rows auto-filled into expected_* (confidence>={min_confidence}): {applied}")


def main():
    parser = argparse.ArgumentParser(description="AI-assisted labeling helper for evaluation manifest.")
    parser.add_argument(
        "--manifest",
        default=str(Path.cwd() / "test_images" / "evaluation_manifest.csv"),
        help="Input manifest CSV path",
    )
    parser.add_argument(
        "--input-root",
        default=str(Path.cwd()),
        help="Root path used to resolve image_path values",
    )
    parser.add_argument(
        "--output",
        default=str(Path.cwd() / "test_images" / "evaluation_manifest_ai_assist.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to process")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum confidence for filling expected_* when --fill-expected is used",
    )
    parser.add_argument(
        "--fill-expected",
        action="store_true",
        help="Fill expected_flood/depth/scene_type from AI suggestions when confidence threshold is met",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    input_root = Path(args.input_root).resolve()
    output_path = Path(args.output).resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    process_manifest(
        manifest_path=manifest_path,
        input_root=input_root,
        output_path=output_path,
        limit=args.limit,
        min_confidence=args.min_confidence,
        fill_expected=args.fill_expected,
    )


if __name__ == "__main__":
    main()
