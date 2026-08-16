"""
AWS-first CLI entrypoint for the flood depth estimator.

This file runs the new AWS/event-driven pipeline by default.
The legacy CLI implementation is preserved in legacy_main.py.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.api_service import FloodApiService


def get_s3_handler(bucket_name: str | None = None):
    """Return the S3 handler for AWS storage."""
    try:
        from archive.legacy_cli.modules.s3_handler import S3Handler
        return S3Handler(bucket_name=bucket_name)
    except ImportError:
        raise RuntimeError("boto3 is required for AWS mode. Install with: pip install boto3")


def read_local_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def read_s3_bytes(s3_handler: Any, s3_key: str) -> bytes:
    response = s3_handler.s3_client.get_object(Bucket=s3_handler.bucket_name, Key=s3_key)
    return response["Body"].read()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_final_table(path: Path, frame: pd.DataFrame, expected_columns: list[str]) -> None:
    if path.exists():
        if path.suffix == ".parquet":
            existing = pd.read_parquet(path)
        else:
            existing = pd.read_csv(path)

        if list(existing.columns) == expected_columns:
            frame = pd.concat([existing, frame], ignore_index=True)
        else:
            backup_path = path.with_name(f"{path.stem}_detailed_backup{path.suffix}")
            if not backup_path.exists():
                path.replace(backup_path)

    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def write_final_prediction_record(result: dict[str, Any], image_name: str | None = None) -> str:
    payload = result.get("result", {}) or {}
    metadata = payload.get("metadata", {}) or {}
    structured = metadata.get("structured_features", {}) or {}
    final_decision = payload.get("final_decision", {}) or {}
    llm_judge = payload.get("llm_judge") or payload.get("llm_judge_result") or {}

    depth_cm = _safe_float(
        final_decision.get("estimated_depth_cm"),
        _safe_float(payload.get("estimated_depth_meters")) * 100.0,
    )
    row = {
        "timestamp": payload.get("processed_at") or payload.get("timestamp"),
        "image_name": image_name or payload.get("image_name"),
        "camera_id": payload.get("camera_id"),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "water_present": bool(payload.get("water_present", depth_cm > 0.0)),
        "final_depth_cm": round(depth_cm, 2),
        "final_severity": final_decision.get("severity_label") or payload.get("severity_label"),
        "final_severity_score": final_decision.get("severity") or payload.get("severity"),
        "final_action": final_decision.get("action_trigger") or payload.get("action_trigger"),
        "final_confidence_pct": round(_safe_float(payload.get("confidence_score")) * 100.0, 2),
        "decision_source": final_decision.get("decision_source") or payload.get("decision_source", "pipeline"),
        "aggregation_source": structured.get("final_aggregation_source"),
        "agreement_status": structured.get("model_agreement_status"),
        "agreement_depth_cm": structured.get("model_agreement_depth_cm"),
        "final_reason": structured.get("final_output_reason"),
        "review_required": bool(payload.get("review_required", structured.get("review_required", False))),
        "review_reason": payload.get("review_reason") or structured.get("review_reason"),
        "llm_status": "available" if llm_judge else "unavailable",
        "llm_recommended_depth_cm": llm_judge.get("recommended_depth_cm") if llm_judge else None,
        "llm_recommended_severity": llm_judge.get("recommended_severity") if llm_judge else None,
    }

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    expected_columns = list(row.keys())

    saved_paths: list[str] = []
    parquet_path = data_dir / "final_predictions.parquet"
    csv_path = data_dir / "final_predictions.csv"

    try:
        _write_final_table(parquet_path, frame, expected_columns)
        saved_paths.append(str(parquet_path))
    except Exception as exc:
        saved_paths.append(f"Parquet unavailable: {exc}")

    _write_final_table(csv_path, frame, expected_columns)
    saved_paths.append(str(csv_path))
    return "; ".join(saved_paths)

def summarize_event_result(result: dict[str, Any], image_name: str | None = None) -> None:
    payload = result.get("result", {})
    metadata = payload.get("metadata", {}) or {}
    structured = metadata.get("structured_features", {}) or {}
    trace = metadata.get("pipeline_trace", []) or []

    final_decision = payload.get("final_decision", {}) or {}
    depth_cm = _safe_float(final_decision.get("estimated_depth_cm"), _safe_float(payload.get("estimated_depth_meters")) * 100.0)
    confidence_pct = float(payload.get("confidence_score", 0.0) * 100.0)
    severity = final_decision.get("severity_label") or payload.get("severity_label", "Unknown")
    action = final_decision.get("action_trigger") or payload.get("action_trigger", "Unknown")
    water_present = bool(payload.get("water_present", depth_cm > 0.0))
    water_coverage = structured.get("water_coverage_pct")
    reference_count = int(structured.get("reference_count", 0))
    reference_depth = structured.get("reference_depth_cm")
    near_water = structured.get("near_water_coverage_pct")
    mid_water = structured.get("mid_water_coverage_pct")
    far_water = structured.get("far_water_coverage_pct")
    model_signals = structured.get("model_signals") or []
    agreement_status = structured.get("model_agreement_status")
    final_reason = structured.get("final_output_reason")

    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("-" * 60)
    print(f"Image: {image_name or '<unknown>'}")
    print(f"Camera: {payload.get('camera_id', 'unknown')}")
    print(f"Water present: {'Yes' if water_present else 'No'}")
    print(f"Final depth: {depth_cm:.2f} cm")
    print(f"Final severity: {severity}")
    print(f"Final action: {action}")
    if final_decision.get("decision_source") or payload.get("decision_source"):
        print(f"Decision source: {final_decision.get('decision_source') or payload.get('decision_source')}")
    print(f"Confidence: {confidence_pct:.2f}%")
    if agreement_status:
        print(f"Model agreement: {agreement_status}")
    if final_reason:
        print(f"Why this final output: {final_reason}")
    print("\nEVIDENCE SUMMARY")
    print("-" * 60)
    if water_coverage is not None:
        print(f"Water coverage: {water_coverage:.2f}%")
    if reference_count is not None:
        print(f"Reference objects found: {reference_count}")
    if reference_depth is not None and reference_depth > 0:
        print(f"Reference-based depth: {reference_depth:.2f} cm")
    if near_water is not None:
        print(f"Near-field water: {near_water:.2f}%")
    if mid_water is not None:
        print(f"Mid-field water: {mid_water:.2f}%")
    if far_water is not None:
        print(f"Far-field water: {far_water:.2f}%")
    print(f"Immediate risk: {bool(structured.get('immediate_risk', False))}")
    print(f"Far water only: {bool(structured.get('far_water_only', False))}")
    print(f"Mask quality warning: {bool(structured.get('mask_quality_warning', False))}")
    if structured.get("full_road_water_no_reference"):
        print("Full-road water without reference object: True")
    if structured.get("low_water_gate_applied"):
        print("Low-water gate applied: True")
        if structured.get("low_water_gate_reason"):
            print(f"Low-water reason: {structured.get('low_water_gate_reason')}")
    if payload.get("review_required"):
        print("Review required: Yes")
        if payload.get("review_reason"):
            print(f"Review reason: {payload.get('review_reason')}")

    if model_signals:
        print("\nModel signals:")
        for signal in model_signals:
            trust = "trusted" if signal.get("trusted") else "not trusted"
            print(f"  - {signal.get('name')}: {signal.get('depth_cm')} cm ({trust})")

    if trace:
        print("\nPipeline summary:")
        for step in trace:
            stage = step.get("stage", "unknown")
            summary = step.get("summary", "")
            print(f" - {stage}: {summary}")

    reference_objects = payload.get("detected_reference_objects") or []
    visual_cues = payload.get("visual_cues") or []
    if reference_objects:
        print("\nReference objects detected:")
        print("  " + ", ".join(reference_objects))
    if visual_cues:
        print("\nVisual evidence cues:")
        for cue in visual_cues:
            print(f"  - {cue}")

    llm_judge = payload.get("llm_judge") or payload.get("llm_judge_result") or metadata.get("llm_judge_result")
    if llm_judge:
        print("\nLLM Judge:")
        print(f"  Prediction correct: {llm_judge.get('prediction_correct', llm_judge.get('plausible'))}")
        print(f"  Recommended depth: {llm_judge.get('recommended_depth_cm')}")
        print(f"  Recommended severity: {llm_judge.get('recommended_severity')}")
        print(f"  Final depth: {payload.get('llm_judge_final_depth_cm', llm_judge.get('final_depth_cm', llm_judge.get('recommended_depth_cm')))}")
        print(f"  Final severity: {payload.get('llm_judge_final_severity', llm_judge.get('final_severity', llm_judge.get('recommended_severity')))}")
        decision_source = payload.get("llm_judge_decision_source") or llm_judge.get("llm_judge_decision_source")
        if decision_source:
            print(f"  Decision source: {decision_source}")
        print(f"  Reason: {llm_judge.get('reason')}")
        if llm_judge.get("parse_failed"):
            print("  Raw response:")
            print(f"    {llm_judge.get('raw_response')}")
    elif payload.get("llm_judge_error"):
        print("\nLLM Judge:")
        print("  Status: unavailable")
        print(f"  Error: {payload.get('llm_judge_error')}")
    saved_output = write_final_prediction_record(result, image_name=image_name)
    print(f"\nSaved final output: {saved_output}")
    print("\nStatus:", result.get("status", "unknown"))
    print("=" * 60 + "\n")


def process_image_cli(
    image_path: str,
    storage_mode: str,
    camera_id: str,
    latitude: float,
    longitude: float,
    location_name: str | None,
    bucket_name: str | None,
) -> None:
    """Analyze a single image through the unified AWS-style pipeline."""
    if storage_mode == "aws":
        s3_handler = get_s3_handler(bucket_name=bucket_name)
        image_bytes = read_s3_bytes(s3_handler, image_path)
    else:
        image_bytes = read_local_bytes(image_path)

    service = FloodApiService()
    response = service.process_camera_upload(
        image_bytes=image_bytes,
        filename=Path(image_path).name,
        camera_id=camera_id,
        latitude=latitude,
        longitude=longitude,
        location_name=location_name,
        metadata={"source": "cli"},
    )

    summarize_event_result(response, image_name=Path(image_path).name)


def process_video_cli(
    video_path: str,
    output_csv: str,
    skip_frames: int,
    storage_mode: str,
    camera_id: str,
    latitude: float,
    longitude: float,
    location_name: str | None,
    bucket_name: str | None,
) -> None:
    """Process video frames through the unified AWS-style pipeline."""
    local_video_path = video_path
    s3_handler = None
    temp_file = None

    if storage_mode == "aws":
        s3_handler = get_s3_handler(bucket_name=bucket_name)
        temp_file = os.path.join(tempfile.gettempdir(), f"aws_video_{Path(video_path).stem}.mp4")
        local_video_path = s3_handler.read_video_from_s3(video_path, temp_file)

    cap = cv2.VideoCapture(str(local_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {local_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Processing video: {local_video_path} ({frame_count} frames, {fps:.2f} FPS)")

    service = FloodApiService()
    records: list[dict[str, Any]] = []
    frame_index = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % skip_frames != 0:
            frame_index += 1
            continue

        processed += 1
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            frame_index += 1
            continue

        image_bytes = encoded.tobytes()
        filename = f"{Path(video_path).stem}_frame_{frame_index:06d}.jpg"
        response = service.process_camera_upload(
            image_bytes=image_bytes,
            filename=filename,
            camera_id=camera_id,
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            metadata={"source": "cli_video", "frame_number": frame_index},
        )

        result = response["result"]
        records.append(
            {
                "frame_number": frame_index,
                "image_name": filename,
                "depth_cm": round(result["estimated_depth_meters"] * 100.0, 2),
                "confidence": round(result["confidence_score"] * 100.0, 2),
                "severity": result["severity_label"],
                "action": result["action_trigger"],
                "status": response["status"],
            }
        )

        if processed % 10 == 0:
            print(f"Processed {processed} frames...")

        frame_index += 1

    cap.release()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved video analytics to {output_csv}")

    if storage_mode == "aws" and s3_handler is not None:
        s3_handler.write_csv_to_s3(df, output_csv)

    if temp_file and os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except OSError:
            pass


def process_object_detection(
    image_path: str,
    output_image: str,
    storage_mode: str,
    bucket_name: str | None,
) -> None:
    """Run object detection on an image. This remains a helper command."""
    from archive.legacy_cli.modules.object_detection import ObjectDetector

    if storage_mode == "aws":
        s3_handler = get_s3_handler(bucket_name=bucket_name)
        image_bytes = read_s3_bytes(s3_handler, image_path)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype="uint8"), cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(f"Cannot read image from {image_path}")

    detector = ObjectDetector()
    detections = detector.detect_objects(image)
    annotated = detector.draw_detections(image, detections)

    if storage_mode == "aws":
        s3_handler.write_image_to_s3(annotated, output_image)
    else:
        cv2.imwrite(output_image, annotated)
        print(f"Saved annotated image to {output_image}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AWS-first flood depth estimator CLI"
    )
    parser.add_argument("mode", nargs="?", choices=["image", "video", "object", "web"], help="Operation mode")
    parser.add_argument("path", nargs="?", help="Path to image or video file")
    parser.add_argument("--storage", choices=["local", "aws"], default="aws", help="Storage mode")
    parser.add_argument("--bucket", help="S3 bucket name when using AWS mode")
    parser.add_argument("--camera-id", default="cli_camera", help="Camera ID for event ingestion")
    parser.add_argument("--latitude", type=float, default=0.0, help="Camera latitude")
    parser.add_argument("--longitude", type=float, default=0.0, help="Camera longitude")
    parser.add_argument("--location-name", help="Camera location name")
    parser.add_argument("--output", help="Output path for CSV or annotated image")
    parser.add_argument("--skip-frames", type=int, default=1, help="Frame skip rate for video")
    parser.add_argument("--app", action="store_true", help="Run the Flask web app")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the Flask app")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Flask app")
    args = parser.parse_args()

    if args.app:
        from web_app import create_app

        app = create_app()
        app.run(host=args.host, port=args.port, debug=True)
        return

    if not args.mode:
        parser.print_help()
        return

    storage_mode = args.storage
    print(f"Using storage mode: {storage_mode}")

    if args.mode == "image":
        if not args.path:
            raise SystemExit("image mode requires a path")
        process_image_cli(
            image_path=args.path,
            storage_mode=storage_mode,
            camera_id=args.camera_id,
            latitude=args.latitude,
            longitude=args.longitude,
            location_name=args.location_name,
            bucket_name=args.bucket,
        )
        return

    if args.mode == "video":
        if not args.path:
            raise SystemExit("video mode requires a path")
        process_video_cli(
            video_path=args.path,
            output_csv=args.output or "video_analysis.csv",
            skip_frames=max(1, args.skip_frames),
            storage_mode=storage_mode,
            camera_id=args.camera_id,
            latitude=args.latitude,
            longitude=args.longitude,
            location_name=args.location_name,
            bucket_name=args.bucket,
        )
        return

    if args.mode == "object":
        if not args.path:
            raise SystemExit("object mode requires a path")
        process_object_detection(
            image_path=args.path,
            output_image=args.output or "objects_detected.jpg",
            storage_mode=storage_mode,
            bucket_name=args.bucket,
        )
        return

    raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
