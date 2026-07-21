# pipeline/temporal.py
"""
Temporal Smoother — per-camera rolling depth filter
=====================================================
Maintains a rolling window of recent depth predictions per camera_id
and applies either an Exponential Moving Average (EMA) or median filter.

Purpose: flood depth readings from a single frame are noisy.  Smoothing
over the last N=5 predictions stabilises the Bengaluru map display and
prevents single-frame spikes from triggering unnecessary alerts.

Config (via pipeline dict or direct kwargs):
  temporal_window_size   int   = 5    rolling window length (median only)
  temporal_smoothing     str   = "ema"  "ema" | "median"
  temporal_alpha         float = 0.3  EMA decay (0=never update, 1=no memory)

Env override: TEMPORAL_SMOOTHING, TEMPORAL_ALPHA, TEMPORAL_WINDOW_SIZE
"""

import logging
from typing import Dict, List

log = logging.getLogger("pipeline.temporal")


class TemporalSmoother:
    """
    Thread-safe-ish in-memory rolling smoother per camera_id.
    State is process-local; restart resets all history (acceptable — the
    queue worker is long-lived, so history accumulates within a deployment).

    Args:
        window_size: number of recent predictions to keep (median mode).
        method:      "ema" — exponential moving average (low lag, recommended)
                     "median" — rolling median (better spike rejection)
        alpha:       EMA decay coefficient.  Larger = faster response to changes.
    """

    def __init__(self, window_size: int = 5, method: str = "ema", alpha: float = 0.3):
        if method not in ("ema", "median"):
            raise ValueError(f"method must be 'ema' or 'median', got {method!r}")
        self.window_size = window_size
        self.method      = method
        self.alpha       = alpha

        self._ema:     Dict[str, float]      = {}
        self._history: Dict[str, List[float]] = {}

    # ── Public API ────────────────────────────────────────────────────────────
    def smooth(self, camera_id: str, depth_cm: float) -> float:
        """
        Accept a new depth reading and return the smoothed value.

        EMA:
          v_new = alpha * raw + (1 - alpha) * v_prev
          First call: initialise to raw value (no warm-up lag).

        Median:
          Return median of the last window_size readings.
        """
        if self.method == "ema":
            return self._ema_smooth(camera_id, depth_cm)
        return self._median_smooth(camera_id, depth_cm)

    def reset(self, camera_id: str | None = None) -> None:
        """Clear history for one camera (or all if camera_id is None)."""
        if camera_id is None:
            self._ema.clear()
            self._history.clear()
        else:
            self._ema.pop(camera_id, None)
            self._history.pop(camera_id, None)

    def state_snapshot(self) -> dict:
        """Return current smoother state — useful for health/debug endpoints."""
        return {
            "method":      self.method,
            "window_size": self.window_size,
            "alpha":       self.alpha,
            "cameras":     len(self._ema) + len(self._history),
        }

    # ── Private ───────────────────────────────────────────────────────────────
    def _ema_smooth(self, camera_id: str, depth_cm: float) -> float:
        if camera_id not in self._ema:
            self._ema[camera_id] = depth_cm
        else:
            self._ema[camera_id] = (
                self.alpha * depth_cm + (1.0 - self.alpha) * self._ema[camera_id]
            )
        return round(self._ema[camera_id], 1)

    def _median_smooth(self, camera_id: str, depth_cm: float) -> float:
        buf = self._history.setdefault(camera_id, [])
        buf.append(depth_cm)
        if len(buf) > self.window_size:
            buf.pop(0)
        # Inline median without numpy dependency (list is small)
        sorted_buf = sorted(buf)
        n = len(sorted_buf)
        mid = n // 2
        if n % 2 == 1:
            median = sorted_buf[mid]
        else:
            median = (sorted_buf[mid - 1] + sorted_buf[mid]) / 2.0
        return round(median, 1)
