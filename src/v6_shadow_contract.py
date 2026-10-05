"""Typed, non-authoritative signal contract for the V6 shadow architecture.

This module intentionally has no dependency on the V5 decision path. A missing or
malformed signal is represented explicitly and is never coerced to numeric zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple


class SignalAuthority(str, Enum):
    PRIMARY_METRIC = "PRIMARY_METRIC"
    CONTEXT_ONLY = "CONTEXT_ONLY"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    ADVISORY_ONLY = "ADVISORY_ONLY"


class SignalValidationError(ValueError):
    """Raised when a present V6 signal violates its declared contract."""


@dataclass(frozen=True)
class Signal:
    name: str
    unit: str
    authority: SignalAuthority
    source_component: str
    value: Optional[Any] = None
    missing_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.value is None:
            if not self.missing_reason:
                object.__setattr__(self, "missing_reason", "not_produced")
            return
        if isinstance(self.value, bool):
            return
        if isinstance(self.value, str):
            raise SignalValidationError(f"{self.name}: strings are not valid present signal values")
        if isinstance(self.value, (int, float)) and not isfinite(float(self.value)):
            raise SignalValidationError(f"{self.name}: non-finite value")

    @property
    def available(self) -> bool:
        return self.value is not None

    @classmethod
    def missing(cls, name: str, unit: str, authority: SignalAuthority, source_component: str, reason: str) -> "Signal":
        return cls(name=name, unit=unit, authority=authority, source_component=source_component, value=None, missing_reason=reason)


@dataclass(frozen=True)
class ObjectDiagnostic:
    object_class: str
    detector_confidence: Signal
    bbox_xyxy: Tuple[int, int, int, int]
    relative_bbox_size: Signal
    vertical_position_ratio: Signal
    submersion_ratio: Signal
    waterline_height_ratio: Signal
    diagnostic_depth_proxy: Signal
    validity_reason: Optional[str] = None

    @property
    def valid_proxy(self) -> bool:
        return self.diagnostic_depth_proxy.available and self.validity_reason is None


@dataclass(frozen=True)
class ObjectAggregateDiagnostics:
    object_count: int
    valid_object_count: int
    diagnostic_proxy_median: Signal
    diagnostic_proxy_dispersion: Signal
    proxy_disagreement_available: bool


@dataclass(frozen=True)
class ReliabilityAssessment:
    semantic_context: Mapping[str, Signal]
    object_diagnostics: ObjectAggregateDiagnostics
    semantic_disagreement_status: str
    object_reference_disagreement_status: str
    efficientnet_context_disagreement: Mapping[str, Signal]
    missing_physical_evidence: Tuple[str, ...]
    backend_fallback_active: bool
    malformed_signal_names: Tuple[str, ...]


@dataclass(frozen=True)
class UncertaintyAssessment:
    reporting_only: bool
    flags: Tuple[str, ...]
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class StageSnapshot:
    stage: str
    numerical_depth_before_cm: Optional[float]
    numerical_depth_after_cm: Optional[float]
    numerical_owner: Optional[str]
    changed_numerical_depth: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class V6SignalContract:
    metric_depth: Mapping[str, Signal]
    semantic_context: Mapping[str, Signal]
    object_diagnostics: Tuple[ObjectDiagnostic, ...]
    object_aggregate: ObjectAggregateDiagnostics
    relative_depth: Mapping[str, Signal]
    advisory: Mapping[str, Signal]
    malformed_signal_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_depth", MappingProxyType(dict(self.metric_depth)))
        object.__setattr__(self, "semantic_context", MappingProxyType(dict(self.semantic_context)))
        object.__setattr__(self, "relative_depth", MappingProxyType(dict(self.relative_depth)))
        object.__setattr__(self, "advisory", MappingProxyType(dict(self.advisory)))

    @property
    def primary_depth_cm(self) -> Optional[float]:
        signal = self.metric_depth["efficientnet_primary_depth_cm"]
        return float(signal.value) if signal.available else None


class FutureSceneType(str, Enum):
    DRY = "DRY"
    WET_NO_FLOOD = "WET_NO_FLOOD"
    DEPTH_0_10 = "0_10_CM"
    DEPTH_10_20 = "10_20_CM"
    DEPTH_20_50 = "20_50_CM"
    DEPTH_50_75 = "50_75_CM"
    DEPTH_75_PLUS = "75_PLUS_CM"


@dataclass(frozen=True)
class FutureShallowTrainingRecord:
    """Schema-only future record. It is not used by current V5 or V6 inference."""
    image_id: str
    scene_type: FutureSceneType
    depth_cm: Optional[float] = None
    label_confidence: Optional[str] = None
    measurement_source: Optional[str] = None
    source_session_id: Optional[str] = None
    physical_reference: Optional[str] = None
