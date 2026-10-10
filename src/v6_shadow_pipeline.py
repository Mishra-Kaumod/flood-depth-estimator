"""V6 single-owner architecture consuming shared primary-model signals.

The primary source loads EfficientNet; a separate shared diagnostic collector
supplies optional evidence. No default path executes legacy depth decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from statistics import median, pstdev
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple
from uuid import uuid4

from .v6_shadow_comparison import V5V6ShadowComparison
from .v6_shadow_contract import (
    ObjectAggregateDiagnostics,
    ObjectDiagnostic,
    ReliabilityAssessment,
    Signal,
    SignalAuthority,
    SignalValidationError,
    StageSnapshot,
    UncertaintyAssessment,
    V6SignalContract,
    CorrectionTrace,
    EvidenceBundle,
    MetricCorrectionCandidate,
)


APPROVED_ABSOLUTE_CEILING_CM = 5.0


def controlled_correction(primary: Optional[float], contract: V6SignalContract,
                          config: Mapping[str, Any] | None = None) -> Tuple[CorrectionTrace, ...]:
    """Sole V6 numerical writer; reject unvalidated, conflicting or oversized proposals.

    Context signals never create centimeters. No candidate is approved in the
    current configuration; this policy can accept only a separately validated
    metric candidate with matching provenance and complete dependencies.
    """
    cfg = dict(config) if isinstance(config, Mapping) else {}
    rejected = [CorrectionTrace(primary, primary, float(signal.value), name, 0.0, False,
                                "unvalidated_diagnostic_candidate", (name,), primary)
                for name, signal in contract.advisory.items()
                if name in ("region_depth_candidate", "mask_conditioned_candidate") and signal.available]
    candidates = contract.metric_candidates
    if not candidates:
        return tuple(rejected) or (CorrectionTrace(primary, primary, None, "controlled_correction_policy", 0.0, False,
                                                 "abstain_no_validated_metric_candidate", (), primary),)

    def reject(candidate: MetricCorrectionCandidate, reason: str) -> CorrectionTrace:
        return CorrectionTrace(primary, primary, candidate.proposed_depth_cm, candidate.source, 0.0,
                               False, reason, candidate.evidence_ids, primary)

    if primary is None or not isfinite(primary) or primary < 0:
        return (*rejected, *(reject(candidate, "primary_unavailable") for candidate in candidates))
    if not cfg.get("enabled", False):
        return (*rejected, *(reject(candidate, "correction_disabled") for candidate in candidates))
    try:
        absolute = float(cfg["max_abs_delta_cm"])
        relative = float(cfg["max_relative_delta_fraction"])
        if not 0 < absolute <= APPROVED_ABSOLUTE_CEILING_CM or not 0 < relative <= 1 or not isfinite(absolute) or not isfinite(relative):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return (*rejected, *(reject(candidate, "invalid_correction_budget") for candidate in candidates))
    approvals = cfg.get("approved_candidates") or []
    if not isinstance(approvals, list):
        return (*rejected, *(reject(candidate, "invalid_approval_registry") for candidate in candidates))
    if len(candidates) != 1:
        return (*rejected, *(reject(candidate, "conflicting_metric_candidates") for candidate in candidates))
    candidate = candidates[0]
    approved = any(isinstance(item, Mapping) and item.get("source") == candidate.source
                   and item.get("validation_id") == candidate.validation_id
                   and candidate.regime in item.get("approved_regimes", ()) for item in approvals)
    if not approved:
        return (*rejected, reject(candidate, "candidate_not_approved_for_regime"))
    collector_status = contract.diagnostic_metadata.get("collector_status", {})
    if any(collector_status.get(name, {}).get("status") != "available" for name in candidate.required_collectors):
        return (*rejected, reject(candidate, "required_collector_unavailable"))
    if contract.malformed_signal_names:
        return (*rejected, reject(candidate, "malformed_supporting_evidence"))
    delta = candidate.proposed_depth_cm - primary
    if delta == 0:
        return (*rejected, reject(candidate, "no_numerical_change"))
    if abs(delta) > min(absolute, relative * primary) + 1e-9:
        return (*rejected, reject(candidate, "proposal_exceeds_correction_budget"))
    return (*rejected, CorrectionTrace(primary, primary, candidate.proposed_depth_cm, candidate.source,
                                       delta, True, "validated_metric_candidate_within_budget",
                                       candidate.evidence_ids, primary + delta))


def assess_eligibility(evidence: Optional[EvidenceBundle], config: Mapping[str, Any]) -> tuple[Optional[bool], str, str]:
    """Evaluate the water gate from water evidence alone, before YOLO runs."""
    if evidence is None:
        return None, "unavailable", "unavailable"
    f, status = evidence.features, evidence.status
    reference = ("none_found" if f.get("reference_count") == 0 else "objects_found") if status.get("yolo", {}).get("status") == "available" else "unavailable"
    if status.get("water", {}).get("status") != "available":
        return None, "uncertain_missing_evidence", reference
    guard = config.get("inference", {}).get("no_water_guard", {})
    coverage, near = f.get("water_coverage_pct"), f.get("near_water_coverage_pct")
    if coverage is None or near is None:
        return None, "uncertain_missing_evidence", reference
    primary = f.get("no_water_probability") if status.get("no_water", {}).get("status") == "available" else None
    wet = f.get("wet_road_no_water_probability") if status.get("wet_road_no_water", {}).get("status") == "available" else None
    # A strong opposite vote is conflict, never a reason to force zero.
    if primary is not None and wet is not None and (
        (primary >= float(guard.get("no_water_threshold", 0.99)) and wet < 1 - float(guard.get("wet_road_guard_threshold", 0.995)))
        or (wet >= float(guard.get("wet_road_guard_threshold", 0.995)) and primary < 1 - float(guard.get("no_water_threshold", 0.99)))
    ):
        return None, "uncertain_conflicting_classifiers", reference
    primary_match = primary is not None and primary >= float(guard.get("no_water_threshold", 0.99)) and coverage <= float(guard.get("max_water_coverage_pct", 5.0)) and near <= float(guard.get("max_near_water_coverage_pct", 5.0))
    wet_match = bool(guard.get("wet_road_guard_enabled", False)) and wet is not None and wet >= float(guard.get("wet_road_guard_threshold", 0.995)) and coverage <= float(guard.get("wet_road_guard_max_water_coverage_pct", 12.0)) and near <= float(guard.get("wet_road_guard_max_near_water_coverage_pct", 8.0))
    if primary_match or wet_match:
        return False, "skipped_corroborated_no_water", reference
    if primary is None and wet is None:
        return None, "uncertain_missing_classifier", reference
    return True, "continue_not_corroborated", reference


@dataclass(frozen=True)
class V6ShadowResult:
    contract: V6SignalContract
    primary_depth_cm: Optional[float]
    final_shadow_depth_cm: Optional[float]
    numerical_owner: str
    reliability: ReliabilityAssessment
    uncertainty: UncertaintyAssessment
    stages: Tuple[StageSnapshot, ...]
    v5_final_depth_cm: Optional[float]
    correction_trace: Tuple[CorrectionTrace, ...] = ()
    evidence: Optional[EvidenceBundle] = None
    water_present: Optional[bool] = None
    water_gate: str = "unavailable"
    reference_eligibility: str = "unavailable"
    depth_inference_skipped: bool = False
    skip_reason: Optional[str] = None
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    model_agreement: Mapping[str, Any] = field(default_factory=dict)

    @property
    def final_v6_depth_cm(self) -> Optional[float]:
        return self.final_shadow_depth_cm

    def comparison(self, actual_depth_cm: Optional[float] = None) -> V5V6ShadowComparison:
        difference = None
        if self.v5_final_depth_cm is not None and self.final_shadow_depth_cm is not None:
            difference = self.final_shadow_depth_cm - self.v5_final_depth_cm
        return V5V6ShadowComparison(
            actual_depth_cm=actual_depth_cm,
            v5_final_depth_cm=self.v5_final_depth_cm,
            v6_primary_depth_cm=self.primary_depth_cm,
            v6_final_shadow_depth_cm=self.final_shadow_depth_cm,
            v6_minus_v5_cm=difference,
            semantic_disagreement=self.reliability.semantic_disagreement_status,
            object_reference_disagreement=self.reliability.object_reference_disagreement_status,
            backend_fallback_active=self.reliability.backend_fallback_active,
            v6_numerical_owner=self.numerical_owner,
        )


class V6ShadowPipeline:
    """V6 numerical owner. It never mutates or configures its signal source."""

    NUMERICAL_OWNER = "efficientnet_primary_anchor"

    def __init__(self, signal_source: Any, evidence_collector: Any = None) -> None:
        self._signal_source = signal_source
        self._evidence_collector = evidence_collector

    @staticmethod
    def _numeric_signal(
        name: str,
        unit: str,
        authority: SignalAuthority,
        source: str,
        value: Any,
        malformed: list[str],
    ) -> Signal:
        if value is None:
            return Signal.missing(name, unit, authority, source, "not_produced")
        if isinstance(value, bool) or isinstance(value, str):
            malformed.append(name)
            return Signal.missing(name, unit, authority, source, "invalid_numeric_type")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            malformed.append(name)
            return Signal.missing(name, unit, authority, source, "invalid_numeric_type")
        if not isfinite(numeric):
            malformed.append(name)
            return Signal.missing(name, unit, authority, source, "nonfinite")
        return Signal(name, unit, authority, source, numeric)

    @staticmethod
    def _bool_signal(name: str, authority: SignalAuthority, source: str, value: Any, malformed: list[str]) -> Signal:
        if value is None:
            return Signal.missing(name, "boolean", authority, source, "not_produced")
        if not isinstance(value, bool):
            malformed.append(name)
            return Signal.missing(name, "boolean", authority, source, "invalid_boolean_type")
        return Signal(name, "boolean", authority, source, value)

    @staticmethod
    def _missing_reason(signal: Signal) -> Optional[str]:
        return signal.missing_reason if not signal.available else None

    def _objects(self, features: Mapping[str, Any], malformed: list[str]) -> Tuple[Tuple[ObjectDiagnostic, ...], ObjectAggregateDiagnostics]:
        height = features.get("input_image_height_px")
        rows = features.get("reference_object_diagnostics") or []
        objects = []
        proxies: list[float] = []
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                malformed.append(f"reference_object_diagnostics[{index}]")
                continue
            bbox_raw = raw.get("bbox")
            try:
                bbox = tuple(int(item) for item in bbox_raw)
                if len(bbox) != 4:
                    raise ValueError
            except (TypeError, ValueError):
                bbox = (0, 0, 0, 0)
                validity_reason = "invalid_bbox"
            else:
                validity_reason = None
            vertical = None
            if height is not None and isinstance(height, (int, float)) and not isinstance(height, bool) and height > 0:
                vertical = ((bbox[1] + bbox[3]) / 2.0) / float(height)
            confidence = self._numeric_signal("detector_confidence", "probability", SignalAuthority.DIAGNOSTIC_ONLY, "yolo_reference_detection", raw.get("detector_confidence"), malformed)
            area = self._numeric_signal("relative_bbox_size", "image_area_ratio", SignalAuthority.DIAGNOSTIC_ONLY, "yolo_reference_detection", raw.get("area_ratio"), malformed)
            vertical_signal = self._numeric_signal("vertical_position_ratio", "image_height_ratio", SignalAuthority.DIAGNOSTIC_ONLY, "yolo_reference_detection", vertical, malformed)
            submersion = self._numeric_signal("submersion_ratio", "ratio", SignalAuthority.DIAGNOSTIC_ONLY, "waterline_overlap", raw.get("water_submersion_ratio"), malformed)
            waterline = self._numeric_signal("waterline_height_ratio", "ratio", SignalAuthority.DIAGNOSTIC_ONLY, "waterline_overlap", raw.get("waterline_height_ratio"), malformed)
            proxy = self._numeric_signal("diagnostic_depth_proxy", "diagnostic_proxy", SignalAuthority.DIAGNOSTIC_ONLY, "nominal_height_x_waterline", raw.get("waterline_depth_proxy_cm"), malformed)
            if proxy.available and validity_reason is None:
                proxies.append(float(proxy.value))
            objects.append(ObjectDiagnostic(
                object_class=str(raw.get("label") or "unknown"), detector_confidence=confidence, bbox_xyxy=bbox,
                relative_bbox_size=area, vertical_position_ratio=vertical_signal, submersion_ratio=submersion,
                waterline_height_ratio=waterline, diagnostic_depth_proxy=proxy, validity_reason=validity_reason,
            ))
        proxy_median = Signal.missing("object_proxy_median", "diagnostic_proxy", SignalAuthority.DIAGNOSTIC_ONLY, "yolo_aggregate", "no_valid_object_proxy")
        proxy_dispersion = Signal.missing("object_proxy_dispersion", "diagnostic_proxy", SignalAuthority.DIAGNOSTIC_ONLY, "yolo_aggregate", "fewer_than_two_valid_object_proxies")
        if proxies:
            proxy_median = Signal("object_proxy_median", "diagnostic_proxy", SignalAuthority.DIAGNOSTIC_ONLY, "yolo_aggregate", float(median(proxies)))
        if len(proxies) >= 2:
            proxy_dispersion = Signal("object_proxy_dispersion", "diagnostic_proxy", SignalAuthority.DIAGNOSTIC_ONLY, "yolo_aggregate", float(pstdev(proxies)))
        return tuple(objects), ObjectAggregateDiagnostics(
            object_count=len(objects), valid_object_count=len(proxies), diagnostic_proxy_median=proxy_median,
            diagnostic_proxy_dispersion=proxy_dispersion, proxy_disagreement_available=len(proxies) >= 2,
        )

    def _contract_from_payload(self, payload: Mapping[str, Any]) -> V6SignalContract:
        features = payload.get("structured_features") or {}
        if not isinstance(features, Mapping):
            raise SignalValidationError("V6 structured_features must be a mapping")
        malformed: list[str] = []
        metric = {
            "efficientnet_primary_depth_cm": self._numeric_signal("efficientnet_primary_depth_cm", "cm", SignalAuthority.PRIMARY_METRIC, "efficientnet_depth_estimator", features.get("efficientnet_candidate_depth_cm"), malformed),
        }
        semantic = {
            name: self._numeric_signal(name, "probability", SignalAuthority.CONTEXT_ONLY, "semantic_classifiers", features.get(name), malformed)
            for name in (
                "road_scene_dry_road_probability", "road_scene_wet_road_probability",
                "road_scene_shallow_flood_probability", "road_scene_meaningful_flood_probability",
                "no_water_probability", "wet_road_no_water_probability", "water_coverage_pct",
                "near_water_coverage_pct", "mid_water_coverage_pct", "far_water_coverage_pct",
            )
        }
        # Coverage is a percentage, not a probability.
        for name in ("water_coverage_pct", "near_water_coverage_pct", "mid_water_coverage_pct", "far_water_coverage_pct"):
            current = semantic[name]
            semantic[name] = Signal(current.name, "percent", current.authority, current.source_component, current.value, current.missing_reason)
        objects, aggregate = self._objects(features, malformed)
        relative = {
            "dense_relative_p90": self._numeric_signal("dense_relative_p90", "relative", SignalAuthority.CONTEXT_ONLY, "depth_anything", features.get("dense_depth_relative_p90"), malformed),
            "dense_relative_map_min": self._numeric_signal("dense_relative_map_min", "relative", SignalAuthority.CONTEXT_ONLY, "depth_anything", features.get("dense_depth_map_min"), malformed),
            "dense_relative_map_max": self._numeric_signal("dense_relative_map_max", "relative", SignalAuthority.CONTEXT_ONLY, "depth_anything", features.get("dense_depth_map_max"), malformed),
            "dense_relative_region_stat": self._numeric_signal("dense_relative_region_stat", "relative", SignalAuthority.CONTEXT_ONLY, "depth_anything", features.get("dense_relative_water_median"), malformed),
        }
        advisory = {
            "contour_reference_depth_estimate": self._numeric_signal("contour_reference_depth_estimate", "cm_estimate", SignalAuthority.DIAGNOSTIC_ONLY, "contour_reference_estimator", features.get("reference_depth_cm"), malformed),
            "region_depth_candidate": self._numeric_signal("region_depth_candidate", "cm_estimate", SignalAuthority.ADVISORY_ONLY, "mask_region_fusion", features.get("region_depth_cm"), malformed),
            "mask_conditioned_candidate": self._numeric_signal("mask_conditioned_candidate", "cm_estimate", SignalAuthority.ADVISORY_ONLY, "mask_conditioned_model", features.get("mask_conditioned_fusion_depth_cm"), malformed),
            "reference_available": self._bool_signal("reference_available", SignalAuthority.DIAGNOSTIC_ONLY, "reference_detection", features.get("reference_available"), malformed),
        }
        metadata = {key: features[key] for key in ("collector_status", "mask_quality", "water_mask_sha256", "semantic_native_predictions",
                    "water_mask_shape", "waterline_image_row_ratio", "object_consistency", "scene_slices",
                    "depth_regime_probabilities", "water_probability", "wet_road_probability",
                    "road_scene_probabilities") if key in features}
        return V6SignalContract(metric, semantic, objects, aggregate, relative, advisory,
                                tuple(sorted(set(malformed))), metadata)

    @staticmethod
    def _reliability(contract: V6SignalContract, payload: Mapping[str, Any]) -> Tuple[ReliabilityAssessment, UncertaintyAssessment]:
        features = payload.get("structured_features") or {}
        primary = contract.metric_depth["efficientnet_primary_depth_cm"]
        region = contract.advisory["region_depth_candidate"]
        reference = contract.advisory["contour_reference_depth_estimate"]
        context_deltas = {}
        for name, signal in (("efficientnet_vs_region", region), ("efficientnet_vs_contour_reference", reference)):
            if primary.available and signal.available:
                context_deltas[name] = Signal(name, "cm_difference", SignalAuthority.DIAGNOSTIC_ONLY, "v6_reliability", abs(float(primary.value) - float(signal.value)))
            else:
                context_deltas[name] = Signal.missing(name, "cm_difference", SignalAuthority.DIAGNOSTIC_ONLY, "v6_reliability", "required_signal_missing")
        missing = []
        if not primary.available:
            missing.append("efficientnet_primary_depth_cm")
        if not contract.object_diagnostics:
            missing.append("yolo_reference_objects")
        dense = contract.relative_depth["dense_relative_p90"]
        if not dense.available:
            missing.append("dense_relative_p90")
        fallback = "proxy" in str(features.get("dense_depth_backend", "")).lower() or "contour" in str(features.get("reference_detection_backend", "")).lower()
        flags = list(contract.malformed_signal_names)
        if missing:
            flags.append("missing_physical_evidence")
        if fallback:
            flags.append("backend_fallback_active")
        votes = features.get("semantic_native_predictions", {})
        scene_vote = votes.get("road_scene")
        guard_votes = [votes[name] for name in ("no_water", "wet_road_no_water") if name in votes]
        disagreement = len(set(guard_votes)) > 1 or (
            scene_vote in ("shallow_flood", "meaningful_flood") and "no_water" in guard_votes) or (
            scene_vote == "dry_road" and "water" in guard_votes)
        if disagreement:
            flags.append("native_semantic_disagreement")
        reliability = ReliabilityAssessment(
            semantic_context=contract.semantic_context, object_diagnostics=contract.object_aggregate,
            semantic_disagreement_status=("native_predictions_disagree" if disagreement else "semantic_probabilities_available_no_metric_authority"
                if any(signal.available for name, signal in contract.semantic_context.items() if "probability" in name)
                else "unavailable"),
            object_reference_disagreement_status="available" if contract.object_aggregate.proxy_disagreement_available else "insufficient_valid_objects",
            efficientnet_context_disagreement=context_deltas, missing_physical_evidence=tuple(missing),
            backend_fallback_active=fallback, malformed_signal_names=contract.malformed_signal_names,
        )
        uncertainty = UncertaintyAssessment(
            reporting_only=True, flags=tuple(sorted(set(flags))),
            notes=("No V6 uncertainty condition changes numerical depth in this phase.",),
        )
        return reliability, uncertainty

    @staticmethod
    def _reference_eligibility(evidence: Optional[EvidenceBundle]) -> str:
        if evidence is None or evidence.status.get("yolo", {}).get("status") != "available":
            return "unavailable"
        count = evidence.features.get("reference_count")
        return "none_found" if count == 0 else "objects_found" if isinstance(count, (int, float)) and count > 0 else "unavailable"

    def _eligibility_result(self, evidence: EvidenceBundle, water_present: Optional[bool], water_gate: str,
                            reference: str, reason: str) -> V6ShadowResult:
        """Return before EfficientNet, agreement, correction, or Gemini can run."""
        payload = {"structured_features": {**dict(evidence.features), "collector_status": dict(evidence.status)}}
        contract = self._contract_from_payload(payload)
        reliability = ReliabilityAssessment(contract.semantic_context, contract.object_aggregate,
                                            "not_run_eligibility_exit", "not_run_eligibility_exit", {}, (), False, ())
        uncertainty = UncertaintyAssessment(True, ("eligibility_exit",), (reason,))
        stages = (StageSnapshot("input", None, None, None, False),
                  StageSnapshot("eligibility_exit", None, None, None, False, {"reason": reason}))
        return V6ShadowResult(contract, None, None, self.NUMERICAL_OWNER, reliability, uncertainty,
                              stages, None, (), evidence, water_present, water_gate, reference,
                              True, reason)

    def predict(self, image_rgb: Any) -> V6ShadowResult:
        """Produce V6 centimetres; only approved metric candidates may change depth."""
        payload = None
        # Eligibility runs before depth and the remaining diagnostic models.
        evidence = None
        config = getattr(self._evidence_collector, "config", {}) or {}
        if not isinstance(config, Mapping):
            config = {}
        eligibility_config = config.get("inference", {}).get("v6_eligibility", {})
        water_present, water_gate, reference_eligibility = None, "unavailable", "unavailable"
        if self._evidence_collector is not None:
            try:
                import numpy as np
                collector = self._evidence_collector
                phased = callable(getattr(type(collector), "collect_water", None))
                if phased:
                    try:
                        evidence = collector.collect_water(np.array(image_rgb, copy=True))
                    except Exception as exc:
                        evidence = EvidenceBundle(status={"water": {"status": "unavailable", "reason": type(exc).__name__}})
                    water_present, water_gate, _ = assess_eligibility(evidence, config)
                    if water_present is False and eligibility_config.get("no_water_zero_enabled", False):
                        return self._eligibility_result(evidence, False, water_gate, "unavailable", "no_flood_water_detected")
                    try:
                        evidence = collector.collect_references(np.array(image_rgb, copy=True), evidence)
                    except Exception as exc:
                        evidence = EvidenceBundle(evidence.features,
                            {**dict(evidence.status), "yolo": {"status": "unavailable", "reason": type(exc).__name__}},
                            evidence.water_mask)
                    reference_eligibility = self._reference_eligibility(evidence)
                    if reference_eligibility == "none_found" and eligibility_config.get("no_reference_na_enabled", False):
                        return self._eligibility_result(evidence, water_present, water_gate, reference_eligibility,
                                                        "no_valid_reference_object_detected")
                    try:
                        evidence = collector.collect_remaining(np.array(image_rgb, copy=True), evidence)
                    except Exception as exc:
                        evidence = EvidenceBundle(evidence.features,
                            {**dict(evidence.status), "remaining": {"status": "unavailable", "reason": type(exc).__name__}},
                            evidence.water_mask)
                else:
                    evidence = collector.collect(np.array(image_rgb, copy=True))
                    water_present, water_gate, _ = assess_eligibility(evidence, config)
                    reference_eligibility = self._reference_eligibility(evidence)
                    if water_present is False and eligibility_config.get("no_water_zero_enabled", False):
                        return self._eligibility_result(evidence, False, water_gate, reference_eligibility,
                                                        "no_flood_water_detected")
                    if reference_eligibility == "none_found" and eligibility_config.get("no_reference_na_enabled", False):
                        return self._eligibility_result(evidence, water_present, water_gate, reference_eligibility,
                                                        "no_valid_reference_object_detected")
                allowed = {"water_coverage_pct", "near_water_coverage_pct", "mid_water_coverage_pct", "far_water_coverage_pct",
                           "road_scene_dry_road_probability", "road_scene_wet_road_probability", "road_scene_shallow_flood_probability",
                           "road_scene_meaningful_flood_probability", "no_water_probability", "wet_road_no_water_probability",
                           "reference_object_diagnostics", "reference_available", "dense_depth_relative_p90", "dense_depth_map_min",
                           "dense_depth_map_max", "dense_relative_water_median", "dense_depth_backend", "reference_detection_backend",
                           "mask_quality", "water_mask_sha256", "water_mask_shape", "waterline_image_row_ratio",
                           "object_consistency", "scene_slices", "semantic_native_predictions", "region_depth_cm", "mask_conditioned_fusion_depth_cm",
                           "reference_count", "no_water_probability", "wet_road_no_water_probability", "water_segmentation_backend",
                           "depth_regime_probabilities", "water_probability", "wet_road_probability", "road_scene_probabilities"}
                features = {k: v for k, v in evidence.features.items() if k in allowed}
                features["collector_status"] = dict(evidence.status)
                payload = {"structured_features": features}
            except Exception as exc:
                evidence = EvidenceBundle(status={"collector": {"status": "unavailable", "reason": type(exc).__name__}})
                payload = {"structured_features": {"collector_status": dict(evidence.status)}}
        if water_present is False:
            water_gate = "confirmed_no_water_control_disabled"
        source_payload = self._signal_source.predict(image_rgb)
        payload = {**source_payload, "structured_features": {**dict(source_payload.get("structured_features") or {}),
                   **dict((payload or {}).get("structured_features") or {})}}
        contract = self._contract_from_payload(payload)
        primary = contract.primary_depth_cm
        correction_config = config.get("inference", {}).get("v6_controlled_correction", {})
        corrections = controlled_correction(primary, contract, correction_config)
        final = corrections[-1].final_v6_depth_cm
        reliability, uncertainty = self._reliability(contract, payload)
        collector_status = contract.diagnostic_metadata.get("collector_status", {})
        model_agreement = {
            "semantic_status": reliability.semantic_disagreement_status,
            "water_evidence_status": collector_status.get("water", {}).get("status", "unavailable"),
            "reference_evidence_status": reference_eligibility,
            "relative_depth_status": collector_status.get("relative_depth", {}).get("status", "unavailable"),
            "metric_candidate_count": len(contract.metric_candidates),
            "correction_supported": any(item.accepted for item in corrections),
            "review_required": reliability.semantic_disagreement_status == "native_predictions_disagree"
                               or water_gate.startswith("uncertain_"),
            "note": "Context agreement does not establish metric correctness",
        }
        stages = (
            StageSnapshot("input", None, None, None, False),
            StageSnapshot("efficientnet_primary_depth", None, primary, self.NUMERICAL_OWNER, primary is not None),
            StageSnapshot("segmentation_context", primary, primary, self.NUMERICAL_OWNER, False, {"water_coverage": contract.semantic_context["water_coverage_pct"].value}),
            StageSnapshot("object_diagnostics", primary, primary, self.NUMERICAL_OWNER, False, {"valid_object_count": contract.object_aggregate.valid_object_count}),
            StageSnapshot("relative_depth_diagnostics", primary, primary, self.NUMERICAL_OWNER, False, {"authority": "CONTEXT_ONLY"}),
            StageSnapshot("semantic_outputs", primary, primary, self.NUMERICAL_OWNER, False, {"authority": "CONTEXT_ONLY"}),
            StageSnapshot("reliability_assessment", primary, primary, self.NUMERICAL_OWNER, False, {"reporting_only": True}),
            StageSnapshot("controlled_correction_policy", primary, final, self.NUMERICAL_OWNER, final != primary,
                          {"mode": "accepted" if any(item.accepted for item in corrections) else "abstained",
                           "proposal_count": len(corrections)}),
            StageSnapshot("uncertainty", final, final, self.NUMERICAL_OWNER, False, {"reporting_only": True}),
            StageSnapshot("final_shadow_output", final, final, self.NUMERICAL_OWNER, False),
        )
        v5_depth = payload.get("depth_cm")
        if v5_depth is not None:
            v5_depth = float(v5_depth)
        return V6ShadowResult(contract, primary, final, self.NUMERICAL_OWNER, reliability, uncertainty, stages, v5_depth,
                              corrections, evidence, water_present, water_gate, reference_eligibility, False,
                              None, model_agreement=model_agreement)
