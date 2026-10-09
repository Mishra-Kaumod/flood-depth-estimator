"""V6 single-owner architecture consuming shared primary-model signals.

The primary source loads EfficientNet; a separate shared diagnostic collector
supplies optional evidence. No default path executes legacy depth decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median, pstdev
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

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
)


def controlled_correction(primary: Optional[float], contract: V6SignalContract) -> Tuple[CorrectionTrace, ...]:
    """SOLE internal correction authority. No secondary candidate is promoted.

    Every proposal abstains until an independently validated acceptance policy
    is explicitly implemented and reviewed. No configurable bypass exists.
    """
    proposals = tuple(CorrectionTrace(primary, primary, float(signal.value), name, 0.0, False,
                                     "shadow_only_no_validated_acceptance_rule", (name,), primary)
                      for name, signal in contract.advisory.items()
                      if name in ("region_depth_candidate", "mask_conditioned_candidate") and signal.available)
    return proposals or (CorrectionTrace(primary, primary, None, "controlled_correction_policy", 0.0, False,
                                         "abstain_no_validated_metric_candidate", (), primary),)


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
                    "water_mask_shape", "waterline_image_row_ratio", "object_consistency", "scene_slices") if key in features}
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

    def predict(self, image_rgb: Any) -> V6ShadowResult:
        """Produce V6 centimetres from primary signals; diagnostics never change depth."""
        payload = self._signal_source.predict(image_rgb)
        payload = {**payload, "structured_features": dict(payload.get("structured_features") or {})}
        # Read primary before diagnostics. Explicit allowlist prevents a collector
        # from smuggling primary/final fields or legacy decisions into V6.
        evidence = None
        if self._evidence_collector is not None:
            try:
                import numpy as np
                evidence = self._evidence_collector.collect(np.array(image_rgb, copy=True))
                allowed = {"water_coverage_pct", "near_water_coverage_pct", "mid_water_coverage_pct", "far_water_coverage_pct",
                           "road_scene_dry_road_probability", "road_scene_wet_road_probability", "road_scene_shallow_flood_probability",
                           "road_scene_meaningful_flood_probability", "no_water_probability", "wet_road_no_water_probability",
                           "reference_object_diagnostics", "reference_available", "dense_depth_relative_p90", "dense_depth_map_min",
                           "dense_depth_map_max", "dense_relative_water_median", "dense_depth_backend", "reference_detection_backend",
                           "mask_quality", "water_mask_sha256", "water_mask_shape", "waterline_image_row_ratio",
                           "object_consistency", "scene_slices", "semantic_native_predictions", "region_depth_cm", "mask_conditioned_fusion_depth_cm"}
                features = dict(payload.get("structured_features") or {})
                features.update({k: v for k, v in evidence.features.items() if k in allowed})
                features["collector_status"] = dict(evidence.status)
                payload = {**payload, "structured_features": features}
            except Exception as exc:
                evidence = EvidenceBundle(status={"collector": {"status": "unavailable", "reason": type(exc).__name__}})
                payload = {**payload, "structured_features": {**payload.get("structured_features", {}), "collector_status": dict(evidence.status)}}
        contract = self._contract_from_payload(payload)
        primary = contract.primary_depth_cm
        corrections = controlled_correction(primary, contract)
        final = corrections[-1].final_v6_depth_cm
        reliability, uncertainty = self._reliability(contract, payload)
        stages = (
            StageSnapshot("input", None, None, None, False),
            StageSnapshot("efficientnet_primary_depth", None, primary, self.NUMERICAL_OWNER, primary is not None),
            StageSnapshot("segmentation_context", primary, primary, self.NUMERICAL_OWNER, False, {"water_coverage": contract.semantic_context["water_coverage_pct"].value}),
            StageSnapshot("object_diagnostics", primary, primary, self.NUMERICAL_OWNER, False, {"valid_object_count": contract.object_aggregate.valid_object_count}),
            StageSnapshot("relative_depth_diagnostics", primary, primary, self.NUMERICAL_OWNER, False, {"authority": "CONTEXT_ONLY"}),
            StageSnapshot("semantic_outputs", primary, primary, self.NUMERICAL_OWNER, False, {"authority": "CONTEXT_ONLY"}),
            StageSnapshot("reliability_assessment", primary, primary, self.NUMERICAL_OWNER, False, {"reporting_only": True}),
            StageSnapshot("controlled_correction_policy", primary, final, self.NUMERICAL_OWNER, final != primary,
                          {"mode": "shadow_abstain", "proposal_count": len(corrections)}),
            StageSnapshot("uncertainty", final, final, self.NUMERICAL_OWNER, False, {"reporting_only": True}),
            StageSnapshot("final_shadow_output", final, final, self.NUMERICAL_OWNER, False),
        )
        v5_depth = payload.get("depth_cm")
        if v5_depth is not None:
            v5_depth = float(v5_depth)
        return V6ShadowResult(contract, primary, final, self.NUMERICAL_OWNER, reliability, uncertainty, stages, v5_depth, corrections, evidence)
