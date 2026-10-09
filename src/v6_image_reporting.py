"""Image-run presentation/export only; prediction values come from shared V6."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any
from uuid import uuid4

IMAGE_REPORT_DIRECTORY = Path("reports/image_predictions")
CSV_FIELDS = (
    "image_filename", "image_path", "prediction_timestamp", "primary_depth_cm",
    "final_shadow_depth_cm", "numerical_owner", "status", "checkpoint_path",
    "checkpoint_sha256", "trace_id", "error_reason", "uncertainty_flags",
    "v6_numerical_owner", "gemini_enabled", "gemini_apply_corrections",
    "gemini_prediction_correct", "gemini_visual_depth_estimate_cm",
    "gemini_visual_depth_range_cm", "gemini_visual_confidence",
    "gemini_recommended_depth_cm", "gemini_final_depth_cm", "gemini_review_required", "gemini_reason",
    "gemini_correction_applied", "gemini_status", "gemini_error_reason",
    "application_final_depth_cm", "decision_source", "diagnostic_evidence",
    "final_v6_depth_cm", "controlled_correction_trace", "gemini_error_code",
)


def build_image_report(image_path: str, payload: dict[str, Any], result: Any, pipeline: Any) -> dict[str, Any]:
    """Copy the already serialized depths; metadata never selects/corrects depth."""
    source = getattr(pipeline, "_signal_source", None)
    backend = getattr(source, "_efficientnet_backend", None)
    checkpoint_path = backend if isinstance(backend, str) else ""
    checkpoint_sha256 = ""
    reasons = []
    if payload["final_shadow_depth_cm"] is None:
        reasons.append("V6 final depth unavailable")
    if checkpoint_path:
        try:
            hasher = hashlib.sha256()
            with Path(checkpoint_path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            checkpoint_sha256 = hasher.hexdigest()
        except OSError as exc:
            reasons.append(f"Checkpoint hash unavailable: {exc}")
    flags = getattr(getattr(result, "uncertainty", None), "flags", ())
    review = payload.get("gemini_review", {})
    return {
        "image_filename": Path(image_path).name,
        "image_path": image_path,
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        **{key: payload[key] for key in ("primary_depth_cm", "final_shadow_depth_cm", "numerical_owner")},
        "v6_numerical_owner": payload["numerical_owner"],
        **{"gemini_" + field: review.get(field) for field in (
            "enabled", "apply_corrections", "prediction_correct", "visual_depth_estimate_cm",
            "visual_depth_range_cm", "visual_confidence", "recommended_depth_cm", "final_depth_cm", "review_required",
            "reason", "correction_applied", "status", "error_reason")},
        "application_final_depth_cm": payload.get("application_final_depth_cm", payload["final_shadow_depth_cm"]),
        "decision_source": payload.get("decision_source", "v6_pipeline"),
        "diagnostic_evidence": json.dumps(payload.get("diagnostic_evidence", {}), allow_nan=False),
        "final_v6_depth_cm": payload.get("final_v6_depth_cm", payload["final_shadow_depth_cm"]),
        "controlled_correction_trace": json.dumps(payload.get("correction_trace", []), allow_nan=False),
        "gemini_error_code": review.get("error_code"),
        "status": "success" if payload["final_shadow_depth_cm"] is not None else "unavailable",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "trace_id": str(uuid4()),
        "error_reason": "; ".join(reasons),
        "uncertainty_flags": json.dumps(list(flags), ensure_ascii=False),
    }


def write_image_report(report: dict[str, Any]) -> Path:
    """Atomically replace one predictable per-image CSV; None becomes empty."""
    directory = IMAGE_REPORT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(report["image_filename"]).stem) or "image"
    output = directory / f"{stem}_v6_prediction.csv"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=directory, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow(report)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def write_image_debug_report(report: dict[str, Any], payload: dict[str, Any], csv_path: Path) -> Path:
    """Write the full sanitized V6 application payload beside the compatible CSV."""
    directory = IMAGE_REPORT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(report["image_filename"]).stem) or "image"
    output = directory / f"{stem}_v6_debug.json"
    document = {"report": report, "csv_report_path": str(csv_path), "result": payload}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _friendly_source(report: dict[str, Any]) -> str:
    source = report.get("decision_source")
    if source == "gemini_review":
        return "Gemini review"
    if source == "v6_pipeline" and report.get("numerical_owner") == "efficientnet_primary_anchor":
        return "EfficientNet"
    return "V6 pipeline"


def _actionable_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    evidence = json.loads(report.get("diagnostic_evidence") or "{}")
    if evidence.get("semantic_disagreement") == "native_predictions_disagree":
        warnings.append("Scene classifiers disagree; review recommended.")
    if report.get("final_shadow_depth_cm") is None:
        warnings.append("V6 returned no usable depth; check the image and configured checkpoint.")

    code = report.get("gemini_error_code")
    warning_by_code = {
        "missing_key": "Gemini review unavailable: API key missing.",
        "authentication": "Gemini review unavailable: API key was rejected; check key access.",
        "quota": "Gemini review unavailable: API quota or rate limit reached.",
        "timeout": "Gemini review unavailable: request timed out.",
        "http_transport": "Gemini review unavailable: network or service error.",
        "blocked_response": "Gemini review unavailable: response was blocked; inspect the image or prompt.",
        "malformed_json": "Gemini review unavailable: response was not valid JSON.",
        "schema_failure": "Gemini review unavailable: response did not match the expected format.",
        "correction_validation_failure": "Gemini correction rejected: recommended depth was missing or invalid.",
    }
    if code in warning_by_code:
        warnings.append(warning_by_code[code])
    elif report.get("gemini_enabled") and report.get("gemini_status") == "failed":
        warnings.append("Gemini review unavailable; V6 depth was retained.")
    return warnings


def print_image_report(report: dict[str, Any], csv_path: Path, payload: dict[str, Any] | None = None,
                       *, verbose: bool = False, debug: bool = False,
                       debug_json_path: Path | None = None) -> None:
    """Print a concise application result; verbose/debug never change prediction values."""
    final_depth = report.get("application_final_depth_cm")
    rendered_depth = f"{final_depth:.2f} cm" if isinstance(final_depth, (int, float)) else "unavailable"
    correction_applied = bool(report.get("gemini_correction_applied"))
    if correction_applied:
        reason = report.get("gemini_reason") or "validated Gemini recommendation accepted"
        correction = f"Applied — {reason[:180]}"
    elif report.get("gemini_enabled") and report.get("gemini_prediction_correct") is False:
        reason = report.get("gemini_reason") or report.get("gemini_error_reason") or "recommendation was not applied"
        correction = f"None — {reason[:180]}"
    else:
        correction = "None — no validated alternative estimate"

    print("Flood Depth Estimator — V6")
    print(f"Image: {report['image_filename']}")
    print()
    print(f"Estimated depth: {rendered_depth}")
    print(f"Depth source: {_friendly_source(report)}")
    print(f"Correction: {correction}")
    print("Inference: completed; estimate is not independently accuracy-verified.")
    warnings = _actionable_warnings(report)
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"\nReport saved: {csv_path}")

    if verbose or debug:
        evidence = json.loads(report.get("diagnostic_evidence") or "{}")
        print("\nDiagnostic summary:")
        for field, label in (("water_coverage_pct", "Water coverage"),
                             ("near_water_coverage_pct", "Near-field water"),
                             ("mid_water_coverage_pct", "Mid-field water"),
                             ("far_water_coverage_pct", "Far-field water"),
                             ("semantic_disagreement", "Scene classifier agreement")):
            value = evidence.get(field)
            if isinstance(value, dict):
                value = value.get("value")
            if value is not None:
                print(f"{label}: {value}")
        trace = json.loads(report.get("controlled_correction_trace") or "[]")
        print(f"Controlled correction proposals: {len(trace)}")
        print(f"Gemini review status: {report.get('gemini_status') or 'disabled'}")
    if debug:
        if payload is not None:
            print("\nComplete application result JSON:")
            print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
        if debug_json_path is not None:
            print(f"Debug report saved: {debug_json_path}")
