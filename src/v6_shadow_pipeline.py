"""Separate V6 shadow architecture built on V5 signal extraction only.

V6 deliberately has one numerical owner in this initial form: the EfficientNet
candidate. V5 may be executed to reuse loaders/extractors, but its final depth,
residual, guards, resolver, and reference fusion never own V6 centimetres.
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
)


class OptionalRefinement(Protocol):
    """Reserved future interface. No current V6 implementation may alter depth here."""

    def apply(self, primary_depth_cm: Optional[float], contract: V6SignalContract) -> Optional[float]: ...


class NoOptionalRefinement:
    """Explicit no-op extension point until a separately trained component is approved."""

    def apply(self, primary_depth_cm: Optional[float], contract: V6SignalContract) -> Optional[float]:
        return primary_depth_cm


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
    """V6 shadow adapter. It never mutates or configures the wrapped V5 pipeline."""

    NUMERICAL_OWNER = "efficientnet_primary_anchor"

    def __init__(self, v5_pipeline: Any) -> None:
        self._v5_pipeline = v5_pipeline
        self.optional_refinement: OptionalRefinement = NoOptionalRefinement()

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

    def _contract_from_v5_payload(self, payload: Mapping[str, Any]) -> V6SignalContract:
        features = payload.get("structured_features") or {}
        if not isinstance(features, Mapping):
            raise SignalValidationError("V5 structured_features must be a mapping")
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
            "dense_relative_region_stat": Signal.missing("dense_relative_region_stat", "relative", SignalAuthority.CONTEXT_ONLY, "depth_anything", "not_available_from_v5_contract"),
        }
        advisory = {
            "contour_reference_depth_estimate": self._numeric_signal("contour_reference_depth_estimate", "cm_estimate", SignalAuthority.DIAGNOSTIC_ONLY, "contour_reference_estimator", features.get("reference_depth_cm"), malformed),
            "region_depth_candidate": self._numeric_signal("region_depth_candidate", "cm_estimate", SignalAuthority.ADVISORY_ONLY, "mask_region_fusion", features.get("region_depth_cm"), malformed),
            "mask_conditioned_candidate": self._numeric_signal("mask_conditioned_candidate", "cm_estimate", SignalAuthority.ADVISORY_ONLY, "mask_conditioned_model", features.get("mask_conditioned_fusion_depth_cm"), malformed),
            "reference_available": self._bool_signal("reference_available", SignalAuthority.DIAGNOSTIC_ONLY, "reference_detection", features.get("reference_available"), malformed),
        }
        return V6SignalContract(metric, semantic, objects, aggregate, relative, advisory, tuple(sorted(set(malformed))))

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
        reliability = ReliabilityAssessment(
            semantic_context=contract.semantic_context, object_diagnostics=contract.object_aggregate,
            semantic_disagreement_status="unassessed_no_calibrated_policy",
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
        """Produce a V6 shadow result while retaining V5 only as a comparator payload."""
        payload = self._v5_pipeline.predict(image_rgb)
        contract = self._contract_from_v5_payload(payload)
        primary = contract.primary_depth_cm
        reliability, uncertainty = self._reliability(contract, payload)
        stages = (
            StageSnapshot("input", None, None, None, False),
            StageSnapshot("segmentation_context", None, None, None, False, {"water_coverage": contract.semantic_context["water_coverage_pct"].value}),
            StageSnapshot("semantic_outputs", None, None, None, False, {"authority": "CONTEXT_ONLY"}),
            StageSnapshot("efficientnet_primary_depth", None, primary, self.NUMERICAL_OWNER, primary is not None),
            StageSnapshot("object_diagnostics", primary, primary, self.NUMERICAL_OWNER, False, {"valid_object_count": contract.object_aggregate.valid_object_count}),
            StageSnapshot("relative_depth_diagnostics", primary, primary, self.NUMERICAL_OWNER, False, {"authority": "CONTEXT_ONLY"}),
            StageSnapshot("reliability_assessment", primary, primary, self.NUMERICAL_OWNER, False, {"reporting_only": True}),
            StageSnapshot("optional_refinement_input", primary, primary, self.NUMERICAL_OWNER, False, {"implementation": "NoOptionalRefinement"}),
            StageSnapshot("optional_refinement_output", primary, primary, self.NUMERICAL_OWNER, False, {"applied": False}),
            StageSnapshot("uncertainty", primary, primary, self.NUMERICAL_OWNER, False, {"reporting_only": True}),
            StageSnapshot("final_shadow_output", primary, primary, self.NUMERICAL_OWNER, False),
        )
        v5_depth = payload.get("depth_cm")
        if v5_depth is not None:
            v5_depth = float(v5_depth)
        return V6ShadowResult(contract, primary, primary, self.NUMERICAL_OWNER, reliability, uncertainty, stages, v5_depth)
