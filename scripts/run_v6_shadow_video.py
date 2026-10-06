#!/usr/bin/env python3
"""Development-only V6 shadow video runner; it does not change the V5 CLI."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from src.v6_video_input import V6VideoInput, VideoFrame


def finite_depth(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def record_from_v6_result(frame_number: int, timestamp_seconds: Optional[float], backend: str, result: Any) -> dict[str, Any]:
    """Read the V6 contract directly; never infer depth from V5 meter fields."""
    primary = finite_depth(result.primary_depth_cm)
    final = finite_depth(result.final_shadow_depth_cm)
    return {
        "frame_index": frame_number,
        "timestamp_sec": timestamp_seconds,
        "saved_frame_path": "",
        "backend_used": backend,
        "primary_depth_cm": primary,
        "final_shadow_depth_cm": final,
        "status": "processed" if final is not None else "unavailable_valid_frame_no_depth",
        "error_reason": "",
    }


CSV_FIELDS = (
    "frame_index", "timestamp_sec", "saved_frame_path", "backend_used",
    "primary_depth_cm", "final_shadow_depth_cm", "status", "error_reason",
)


def load_saved_rgb(path: Path) -> np.ndarray:
    """Match the single-image V6 comparison path: PIL RGB after saving."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def process_saved_frames(
    reader: V6VideoInput,
    v6_pipeline: Any,
    video_path: str,
    run_directory: Path,
    max_frames: Optional[int],
    skip_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Save each decoded frame, reload it, then invoke the V6 image contract."""
    frames_directory = run_directory / "frames"
    frames_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    decoded_count = 0
    for frame in reader.iter_frames(video_path):
        if max_frames is not None and frame.frame_index >= max_frames:
            break
        decoded_count += 1
        if frame.frame_index % skip_frames:
            continue
        saved_path = frames_directory / f"frame_{frame.frame_index:06d}.jpg"
        try:
            if not cv2.imwrite(str(saved_path), frame.frame_bgr):
                raise RuntimeError("cv2.imwrite returned false")
            result = v6_pipeline.predict(load_saved_rgb(saved_path))
            row = record_from_v6_result(frame.frame_index, frame.timestamp_seconds, frame.backend, result)
            row["saved_frame_path"] = str(saved_path)
            rows.append(row)
        except Exception as exc:  # A bad saved frame or image inference must not abort the video.
            rows.append({
                "frame_index": frame.frame_index, "timestamp_sec": frame.timestamp_seconds,
                "saved_frame_path": str(saved_path) if saved_path.exists() else "",
                "backend_used": frame.backend, "primary_depth_cm": None,
                "final_shadow_depth_cm": None, "status": "skipped_bad_frame",
                "error_reason": str(exc),
            })
    for diagnostic in reader.diagnostics:
        rows.append({
            "frame_index": diagnostic.frame_index, "timestamp_sec": None, "saved_frame_path": "",
            "backend_used": reader.backend or "none", "primary_depth_cm": None,
            "final_shadow_depth_cm": None, "status": diagnostic.code,
            "error_reason": diagnostic.message,
        })
    summary = {
        "video_path": str(video_path), "decoded_frame_count": decoded_count,
        "processed_record_count": len(rows), "backend_used": reader.backend,
        "fps": reader.fps, "max_frames": max_frames, "skip_frames": skip_frames,
        "diagnostics": [{"code": item.code, "message": item.message, "frame_index": item.frame_index} for item in reader.diagnostics],
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="V6-only shadow video processing")
    parser.add_argument("--video", required=True, help="Explicit local MP4/DAV/DHAV input path")
    parser.add_argument("--output-dir", help="Run directory; defaults to reports/v6_video_runs/<video_name>")
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int, help="Optional cap on decoded frames for a bounded smoke test")
    args = parser.parse_args()
    if args.skip_frames < 1:
        raise ValueError("--skip-frames must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")

    run_directory = Path(args.output_dir) if args.output_dir else Path("reports") / "v6_video_runs" / Path(args.video).stem
    run_directory.mkdir(parents=True, exist_ok=True)
    # Imports stay here so tests can exercise the saved-frame flow without loading models.
    from src.segformer_yolo_depthv2_pipeline import SegformerYoloDepthV2Pipeline
    from src.v6_shadow_pipeline import V6ShadowPipeline

    reader = V6VideoInput()
    rows, summary = process_saved_frames(reader, V6ShadowPipeline(SegformerYoloDepthV2Pipeline()), args.video, run_directory, args.max_frames, args.skip_frames)
    csv_path = run_directory / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS); writer.writeheader(); writer.writerows(rows)
    summary_path = run_directory / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_directory": str(run_directory), "predictions": str(csv_path), "summary": str(summary_path), "backend": reader.backend, "fps": reader.fps}))


if __name__ == "__main__":
    main()
