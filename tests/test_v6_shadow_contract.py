import math
import unittest

from src.v6_shadow_contract import Signal, SignalAuthority, SignalValidationError


class V6SignalContractTests(unittest.TestCase):
    def test_missing_signal_stays_missing(self):
        signal = Signal.missing("primary", "cm", SignalAuthority.PRIMARY_METRIC, "efficientnet", "backend_unavailable")
        self.assertFalse(signal.available)
        self.assertIsNone(signal.value)
        self.assertEqual(signal.missing_reason, "backend_unavailable")

    def test_nan_is_rejected(self):
        with self.assertRaises(SignalValidationError):
            Signal("primary", "cm", SignalAuthority.PRIMARY_METRIC, "efficientnet", math.nan)

    def test_string_boolean_is_rejected(self):
        with self.assertRaises(SignalValidationError):
            Signal("reference_available", "boolean", SignalAuthority.DIAGNOSTIC_ONLY, "yolo", "false")

    def test_relative_depth_is_not_centimetres(self):
        signal = Signal("dense_relative_p90", "relative", SignalAuthority.CONTEXT_ONLY, "depth_anything", 0.8)
        self.assertEqual(signal.unit, "relative")
        self.assertNotIn("cm", signal.unit)
