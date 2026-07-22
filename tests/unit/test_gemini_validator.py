# tests/unit/test_gemini_validator.py
"""
Unit tests for GeminiValidator — ensemble logic and anchoring-bias fix.
Run: pytest tests/unit/test_gemini_validator.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.gemini_validator import GeminiValidator, GeminiResult, EnsembleResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_gemini_result(depth_cm: float, risk: str, confidence_pct: float) -> GeminiResult:
    return GeminiResult(
        flood_detected=depth_cm > 0,
        estimated_depth_cm=depth_cm,
        risk_level=risk,
        confidence_pct=confidence_pct,
        reasoning="test",
        raw_response="{}",
        success=True,
    )

def run_ensemble(model_depth, model_risk, model_conf, gemini_depth, gemini_risk, gemini_conf):
    """Drive _ensemble() directly without an API key."""
    validator = GeminiValidator(api_key="dummy")
    gemini = make_gemini_result(gemini_depth, gemini_risk, gemini_conf)
    return validator._ensemble(gemini, model_depth, model_risk, model_conf)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEnsembleWeighting:

    def test_high_model_confidence_weights_toward_model(self):
        """
        model_conf=0.9, gemini_conf=10% → w_model≈0.9
        Ensemble depth should be much closer to model_depth than gemini_depth.
        """
        r = run_ensemble(
            model_depth=50, model_risk="HIGH RISK", model_conf=0.9,
            gemini_depth=10, gemini_risk="LOW RISK",  gemini_conf=10,   # 10%
        )
        assert r.w_model > 0.8, f"Expected w_model > 0.8, got {r.w_model}"
        assert r.water_depth_cm > 40, f"Expected depth > 40 cm, got {r.water_depth_cm}"

    def test_high_gemini_confidence_weights_toward_gemini(self):
        """
        model_conf=0.2, gemini_conf=90% → w_gemini≈0.82
        Ensemble depth should be much closer to gemini_depth than model_depth.
        """
        r = run_ensemble(
            model_depth=10, model_risk="LOW RISK",  model_conf=0.2,
            gemini_depth=60, gemini_risk="HIGH RISK", gemini_conf=90,   # 90%
        )
        assert r.w_gemini > 0.7, f"Expected w_gemini > 0.7, got {r.w_gemini}"
        assert r.water_depth_cm > 45, f"Expected depth > 45 cm, got {r.water_depth_cm}"

    def test_equal_confidence_gives_equal_weights(self):
        r = run_ensemble(
            model_depth=20, model_risk="MODERATE", model_conf=0.5,
            gemini_depth=40, gemini_risk="MODERATE", gemini_conf=50,   # 50%
        )
        assert abs(r.w_model - r.w_gemini) < 0.01, (
            f"Equal conf should give equal weights: w_model={r.w_model} w_gemini={r.w_gemini}"
        )
        assert abs(r.water_depth_cm - 30.0) < 0.5, f"Expected ~30cm, got {r.water_depth_cm}"

    def test_weights_always_sum_to_one(self):
        for mc, gc in [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (0.0, 0.8), (0.8, 0.0)]:
            r = run_ensemble(20, "MODERATE", mc, 40, "HIGH RISK", gc * 100)
            assert abs(r.w_model + r.w_gemini - 1.0) < 1e-6, (
                f"Weights must sum to 1: w_model={r.w_model}, w_gemini={r.w_gemini}"
            )

    def test_zero_confidence_both_gives_50_50(self):
        r = run_ensemble(20, "MODERATE", 0.0, 40, "HIGH RISK", 0.0)
        assert abs(r.w_model - 0.5) < 0.01, f"Expected 0.5/0.5, got {r.w_model}"

    def test_safety_first_risk_takes_higher(self):
        """Even when model says LOW RISK and Gemini says HIGH RISK, ensemble = HIGH RISK."""
        r = run_ensemble(10, "LOW RISK", 0.9, 50, "HIGH RISK", 0.3)
        assert r.risk_level == "HIGH RISK", f"Expected HIGH RISK, got {r.risk_level}"

    def test_agreement_flag_set_correctly(self):
        agree = run_ensemble(30, "MODERATE", 0.8, 32, "MODERATE", 0.7)
        disagree = run_ensemble(30, "MODERATE", 0.8, 70, "HIGH RISK", 0.7)
        assert agree.agreement is True
        assert disagree.agreement is False

    def test_ensemble_method_label(self):
        r = run_ensemble(20, "MODERATE", 0.7, 25, "MODERATE", 0.6)
        assert r.ensemble_method == "model_gemini_weighted"

    def test_depth_cap_no_longer_120(self):
        """Gemini should be able to report >120cm and it flows through unchanged."""
        r = run_ensemble(
            model_depth=150, model_risk="CRITICAL", model_conf=0.6,
            gemini_depth=180, gemini_risk="CRITICAL", gemini_conf=0.6,
        )
        assert r.water_depth_cm > 120, f"Expected >120cm, got {r.water_depth_cm}"
