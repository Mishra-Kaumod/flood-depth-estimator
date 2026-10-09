"""Optional application review AFTER immutable shared V6 inference.

This module owns Gemini orchestration and application output only. It never
loads models, preprocesses images or changes the V6 result/serializer.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Callable

from src.llm_judge import LLMJudge, GeminiReviewError
from src.settings import load_settings_dict
from src.v6_inference import finite_depth, v6_depth_payload

GEMINI_FIELDS = (
    "prediction_correct", "visual_depth_estimate_cm", "visual_depth_range_cm",
    "visual_confidence", "recommended_depth_cm", "review_required", "reason",
    "final_depth_cm",
)


def diagnostic_evidence(result: Any) -> dict[str, Any]:
    """Export only available typed context; never run missing legacy components."""
    contract = getattr(result, "contract", None)
    if contract is None:
        return {}
    evidence = {}
    for group in (contract.semantic_context, contract.relative_depth, contract.advisory):
        for name, signal in group.items():
            if name == "contour_reference_depth_estimate":
                continue
            if signal.available:
                evidence[name] = {"value": signal.value, "unit": signal.unit,
                                  "authority": signal.authority.value}
    yolo_available = getattr(contract, "diagnostic_metadata", {}).get("collector_status", {}).get("yolo", {}).get("status") == "available"
    if contract.object_diagnostics or yolo_available:
        evidence["reference_objects"] = [{key: value for key, value in asdict(obj).items()
                                           if key != "diagnostic_depth_proxy"} for obj in contract.object_diagnostics]
        evidence["reference_object_count"] = len(contract.object_diagnostics)
    evidence.update(dict(getattr(contract, "diagnostic_metadata", {})))
    evidence["semantic_disagreement"] = result.reliability.semantic_disagreement_status
    # Current default source does not supply visual cues or reference objects.
    # Missing evidence stays absent; it must not be reconstructed from V5.
    return json.loads(json.dumps(evidence, allow_nan=False))


def _valid_depth(value: Any, maximum: float | None) -> float | None:
    if isinstance(value, (bool, list, dict)):
        return None
    depth = finite_depth(value)
    if depth is None or depth < 0 or (maximum is not None and depth > maximum):
        return None
    return depth


def review_v6_result(result: Any, image_bytes: bytes | Callable[[], bytes], filename: str,
                     pipeline: Any = None, *, video: bool = False,
                     config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Single application boundary reused by CLI, UI, event/API and saved frames.

    Caller completes V6 first. Review failures are diagnostic only. Corrections
    require explicit disagreement plus a valid nonnegative bounded depth.
    """
    raw = v6_depth_payload(result)
    output = {**raw, "application_final_depth_cm": raw["final_v6_depth_cm"],
              "decision_source": "v6_pipeline", "diagnostic_evidence": {},
              "gemini_review": {"enabled": False, "apply_corrections": False,
                                "correction_applied": False, "status": "disabled",
                                **{field: None for field in GEMINI_FIELDS}, "error_reason": None, "error_code": None}}
    output["correction_trace"] = [{**asdict(item), "evidence_ids": list(item.evidence_ids)} for item in getattr(result, "correction_trace", ())]
    output["gemini_recommended_depth_cm"] = None
    review = output["gemini_review"]
    try:
        output["diagnostic_evidence"] = diagnostic_evidence(result)
        settings = config if config is not None else load_settings_dict()
        inference = settings.get("inference", {})
        cfg = inference.get("llm_judge", {})
        review["apply_corrections"] = bool(cfg.get("apply_corrections", False))
        review["enabled"] = bool(cfg.get("enabled", False)) and (
            not video or bool(inference.get("video", {}).get("gemini_review_enabled", False)))
        if not review["enabled"]:
            return output
        if raw["final_v6_depth_cm"] is None:
            review.update(status="skipped_unavailable_v6", error_reason="V6 depth unavailable; review skipped")
            return output
        judge = LLMJudge(cfg)
        # Loaded checkpoint scale is authoritative; config scale is a fallback
        # for callers without a model handle. No clamp changes the V6 result.
        source = getattr(pipeline, "_signal_source", None)
        bound = getattr(source, "_efficientnet_max_depth_cm", None)
        if bound is None:
            bound = inference.get("efficientnet_signal", {}).get("max_depth_cm")
        maximum = finite_depth(bound)
        if bound is not None and (maximum is None or maximum <= 0):
            raise GeminiReviewError("correction_validation_failure")
        parsed = judge.judge_v6(
            {"v6_metric_prediction": raw, "diagnostic_context": output["diagnostic_evidence"],
             "controlled_correction_trace": output["correction_trace"]},
            image_bytes=image_bytes() if callable(image_bytes) else image_bytes, filename=filename)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("prediction_correct"), bool):
            raise GeminiReviewError("schema_failure")
        # Whitelist diagnostics, omit raw responses and legacy severity fields.
        for field in GEMINI_FIELDS:
            value = parsed.get(field)
            if field.endswith("_cm") and field != "visual_depth_range_cm":
                value = _valid_depth(value, maximum)
            elif field in ("prediction_correct", "review_required"):
                value = value if isinstance(value, bool) else None
            elif field == "visual_confidence":
                value = value if value in ("low", "medium", "high") else None
            else:
                value = value[:2000] if isinstance(value, str) else None
                if value and judge.api_key:
                    value = value.replace(str(judge.api_key), "[redacted]")
            review[field] = value
        output["gemini_recommended_depth_cm"] = review["recommended_depth_cm"]
        candidate = parsed.get("final_depth_cm") if parsed.get("final_depth_cm") is not None else parsed.get("recommended_depth_cm")
        corrected = _valid_depth(candidate, maximum)
        if parsed["prediction_correct"] is False and (corrected is None or maximum is None):
            review.update(status="invalid_depth", error_code="correction_validation_failure", error_reason="Gemini correction depth missing or invalid")
            return output
        review["status"] = "reviewed"
        if review["apply_corrections"] and parsed["prediction_correct"] is False:
            output.update(application_final_depth_cm=corrected, decision_source="gemini_review")
            review["correction_applied"] = True
    except Exception as exc:
        # Do not expose exception bodies/URLs/headers, which can contain secrets.
        code = exc.code if isinstance(exc, GeminiReviewError) else (
            "timeout" if isinstance(exc, TimeoutError) else "http_transport" if isinstance(exc, OSError) else "schema_failure")
        review.update(status="failed", error_code=code, error_reason=f"Gemini review unavailable ({code}); V6 retained")
    return output
