"""Development-only V5/V6 comparison objects; no decision logic lives here."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class V5V6ShadowComparison:
    actual_depth_cm: Optional[float]
    v5_final_depth_cm: Optional[float]
    v6_primary_depth_cm: Optional[float]
    v6_final_shadow_depth_cm: Optional[float]
    v6_minus_v5_cm: Optional[float]
    semantic_disagreement: str
    object_reference_disagreement: str
    backend_fallback_active: bool
    v6_numerical_owner: str

    def as_dict(self) -> Mapping[str, Any]:
        return asdict(self)
