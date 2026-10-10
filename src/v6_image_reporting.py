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
    "water_present", "water_gate", "reference_eligibility", "depth_inference_skipped", "skip_reason",
    "prediction_status", "comment",
    "v6_review_required",
    "model_agreement", "correction_proposed_depth_cm", "correction_accepted",
    "correction_rejected_reason", "accepted_delta_cm",
)


def build_image_report(image_path: str, payload: dict[str, Any], result: Any, pipeline: Any) -> dict[str, Any]:
    """Copy the already serialized depths; metadata never selects/corrects depth."""
    source = getattr(pipeline, "_signal_source", None)
    backend = getattr(source, "_efficientnet_backend", None)
    checkpoint_path = backend if isinstance(backend, str) else ""
    checkpoint_sha256 = ""
    reasons = []
    if payload["final_shadow_depth_cm"] is None and not payload.get("depth_inference_skipped"):
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
    trace = payload.get("correction_trace", [])
    accepted = next((item for item in trace if item.get("accepted")), None)
    proposed = next((item for item in trace if item.get("proposed_depth_cm") is not None), None)
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
        **{key: payload.get(key) for key in ("water_present", "water_gate", "reference_eligibility", "depth_inference_skipped", "skip_reason", "prediction_status", "comment")},
        "v6_review_required": payload.get("v6_review_required"),
        "model_agreement": json.dumps((payload.get("diagnostic_evidence") or {}).get("model_agreement", {}), allow_nan=False),
        "correction_proposed_depth_cm": proposed.get("proposed_depth_cm") if proposed else None,
        "correction_accepted": bool(accepted),
        "correction_rejected_reason": None if accepted else proposed.get("acceptance_or_rejection_reason") if proposed else "no_validated_metric_candidate",
        "accepted_delta_cm": accepted.get("correction_amount_cm") if accepted else 0.0,
        "status": "no_flood_detected" if payload.get("skip_reason") == "no_flood_water_detected" else
                  "skipped_no_reference" if payload.get("skip_reason") == "no_valid_reference_object_detected" else
                  "success" if payload["final_shadow_depth_cm"] is not None else "unavailable",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "trace_id": getattr(result, "trace_id", None) or str(uuid4()),
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
    if source == "no_water_gate":
        return "Confirmed no-water gate"
    if source == "no_reference_gate":
        return "Reference eligibility"
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
    if report.get("final_shadow_depth_cm") is None and not report.get("depth_inference_skipped"):
        warnings.append("V6 returned no usable depth; check the image and configured checkpoint.")

    code = report.get("gemini_error_code")
    warning_by_code = {
        "missing_key": "Gemini review unavailable: API key missing.",
        "authentication": "Gemini review unavailable: API key was rejected; check key access.",
        "quota": "Gemini review unavailable: API quota or rate limit reached.",
        "timeout": "Gemini review unavailable: request timed out.",
        "http_transport": "Gemini review unavailable: network or service error.",
        "model_endpoint": "Gemini review unavailable: configured model or endpoint was not found.",
        "request_payload": "Gemini review unavailable: request payload was rejected or invalid.",
        "configuration": "Gemini review unavailable: check reviewer configuration.",
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
    rendered_depth = (f"{final_depth:.2f} cm" if isinstance(final_depth, (int, float)) else
                      "N/A" if report.get("skip_reason") == "no_valid_reference_object_detected" else "unavailable")
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
    print(f"Status: {report.get('prediction_status') or 'unavailable'}")
    print(f"Comment: {report.get('comment') or 'unavailable'}")
    print(f"Depth source: {_friendly_source(report)}")
    print(f"Decision source: {report.get('decision_source') or 'unavailable'}")
    print(f"Correction: {correction}")
    print(f"Water gate: {report.get('water_gate') or 'unavailable'}")
    print(f"Reference eligibility: {report.get('reference_eligibility') or 'unavailable'}")
    if report.get("depth_inference_skipped"):
        print(f"Depth inference skipped: {report.get('skip_reason')}")
    else:
        print("Inference: completed; estimate is not independently accuracy-verified.")
    warnings = _actionable_warnings(report)
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"\nReport saved: {csv_path}")

    if verbose or debug:
        evidence = json.loads(report.get("diagnostic_evidence") or "{}")
        def shown(value):
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            return "unavailable" if value is None else str(value)
        print("\nEligibility\n-----------")
        print(f"Water present: {shown(report.get('water_present'))}")
        print(f"Water gate: {shown(report.get('water_gate'))}")
        print(f"Reference eligibility: {shown(report.get('reference_eligibility'))}")
        print(f"Depth inference skipped: {bool(report.get('depth_inference_skipped'))}")
        print(f"Skip reason: {shown(report.get('skip_reason'))}")
        print("\nV6 Primary Prediction\n---------------------")
        print(f"EfficientNet depth: {shown(report.get('primary_depth_cm'))} cm")
        print(f"Numerical owner: {shown(report.get('numerical_owner'))}")
        print("\nDiagnostic Evidence\n-------------------")
        for field, label in (("water_coverage_pct", "Water coverage"),
                             ("near_water_coverage_pct", "Near-field water"),
                             ("mid_water_coverage_pct", "Mid-field water"),
                             ("far_water_coverage_pct", "Far-field water"),
                             ("wet_road_probability", "Wet-road probability"),
                             ("water_probability", "Water probability"),
                             ("no_water_probability", "No-water probability"),
                             ("reference_object_count", "YOLO objects"),
                             ("dense_relative_p90", "Depth Anything relative"),
                             ("semantic_disagreement", "Scene classifier agreement")):
            value = evidence.get(field)
            print(f"{label}: {shown(value)}")
        objects = evidence.get("reference_objects") or []
        print("Reference evidence: " + (", ".join(str(item.get("object_class", "object")) for item in objects[:8]) if objects else "unavailable or none found"))
        print(f"Road-scene probabilities: {shown(evidence.get('road_scene_probabilities'))}")
        print(f"Depth-regime probabilities: {shown(evidence.get('depth_regime_probabilities'))}")
        print(f"Experimental candidates: {shown((evidence.get('collector_status') or {}).get('experimental_candidates'))}")
        agreement = evidence.get("model_agreement") or {}
        print("\nModel Agreement\n---------------")
        print(f"Water evidence: {shown(agreement.get('water_evidence_status'))}")
        print(f"YOLO reference evidence: {shown(agreement.get('reference_evidence_status'))}")
        print(f"Semantic agreement/disagreement: {shown(agreement.get('semantic_status'))}")
        print(f"Validated metric candidate count: {shown(agreement.get('metric_candidate_count'))}")
        print(f"Review required: {shown(report.get('v6_review_required'))}")
        trace = json.loads(report.get("controlled_correction_trace") or "[]")
        print("\nControlled V6 Decision\n----------------------")
        print(f"Proposals: {len(trace)}")
        for proposal in trace:
            print(f"- {proposal.get('correction_source')}: proposed {shown(proposal.get('proposed_depth_cm'))} cm; {proposal.get('acceptance_or_rejection_reason')}")
        print(f"Accepted correction: {any(item.get('accepted') for item in trace)}")
        print(f"Accepted correction amount: {sum(item.get('correction_amount_cm', 0) for item in trace if item.get('accepted'))} cm")
        print(f"Final V6 depth: {shown(report.get('final_v6_depth_cm'))} cm")
        print("\nGemini Review\n-------------")
        for key, label in (("gemini_status", "Status"), ("gemini_prediction_correct", "Prediction correct"),
                           ("gemini_visual_depth_estimate_cm", "Visual estimate"),
                           ("gemini_visual_depth_range_cm", "Visual range"),
                           ("gemini_visual_confidence", "Visual confidence"),
                           ("gemini_recommended_depth_cm", "Recommended depth"),
                           ("gemini_review_required", "Review required"),
                           ("gemini_correction_applied", "Correction applied"), ("gemini_reason", "Reason")):
            print(f"{label}: {shown(report.get(key))}")
        print("\nFinal Application Result\n------------------------")
        print(f"V6 final: {shown(report.get('final_v6_depth_cm'))} cm")
        print(f"Application final: {shown(report.get('application_final_depth_cm'))} cm")
        print(f"Decision source: {shown(report.get('decision_source'))}")
    if debug:
        if payload is not None:
            print("\nComplete application result JSON:")
            print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
        if debug_json_path is not None:
            print(f"Debug report saved: {debug_json_path}")
