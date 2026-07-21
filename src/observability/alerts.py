# src/observability/alerts.py
"""
Alert Engine
=============
Fires when flood severity reaches the configured threshold (HIGH RISK / CRITICAL).

Channels supported:
  1. Slack / MS Teams  — webhook POST (JSON)
  2. SMS               — Twilio
  3. Generic webhook   — any HTTP endpoint

Designed as a fire-and-forget background thread so it never blocks the pipeline.
"""

import json
import logging
import threading
import time
import urllib.request
from datetime import datetime
from typing import Optional

log = logging.getLogger("observability.alerts")

RISK_ORDER = ["NO FLOOD", "LOW RISK", "MODERATE", "HIGH RISK", "CRITICAL"]

RISK_EMOJI = {
    "MODERATE":  "🟠",
    "HIGH RISK": "🔴",
    "CRITICAL":  "🚨",
}


def _risk_gte(risk: str, min_risk: str) -> bool:
    try:
        return RISK_ORDER.index(risk) >= RISK_ORDER.index(min_risk)
    except ValueError:
        return False


class AlertEngine:
    """
    Receives FloodPrediction objects.
    Fires alerts asynchronously when risk >= alert_min_risk.

    Thread-safe. Deduplicates: won't re-alert same camera within cooldown_s.
    """

    def __init__(
        self,
        enabled:         bool  = False,
        min_risk:        str   = "HIGH RISK",
        webhook_url:     str   = "",
        twilio_sid:      str   = "",
        twilio_token:    str   = "",
        twilio_from:     str   = "",
        sms_numbers:     list[str] = None,
        cooldown_s:      int   = 1800,   # 30 min between alerts for same camera
    ):
        self.enabled     = enabled
        self.min_risk    = min_risk
        self.webhook_url = webhook_url
        self.twilio_sid  = twilio_sid
        self.twilio_token= twilio_token
        self.twilio_from = twilio_from
        self.sms_numbers = sms_numbers or []
        self.cooldown_s  = cooldown_s
        self._last_alert: dict[str, float] = {}   # camera_id → last alert timestamp
        self._lock = threading.Lock()

        if enabled:
            log.info("Alert engine enabled — min_risk=%s channels: %s",
                     min_risk,
                     ", ".join(filter(None, [
                         "webhook" if webhook_url else "",
                         "sms"     if twilio_sid  else "",
                     ])) or "none configured")

    # ── Public ────────────────────────────────────────────────────────────────
    def check_and_fire(self, prediction) -> bool:
        """
        Call after every pipeline prediction.
        Returns True if an alert was fired.
        """
        if not self.enabled:
            return False
        if not _risk_gte(prediction.risk_level, self.min_risk):
            return False
        if self._in_cooldown(prediction.camera_id):
            return False

        self._mark_alerted(prediction.camera_id)
        threading.Thread(
            target=self._dispatch_all, args=(prediction,), daemon=True
        ).start()
        return True

    # ── Internal ──────────────────────────────────────────────────────────────
    def _in_cooldown(self, camera_id: str) -> bool:
        with self._lock:
            last = self._last_alert.get(camera_id, 0)
            return (time.time() - last) < self.cooldown_s

    def _mark_alerted(self, camera_id: str):
        with self._lock:
            self._last_alert[camera_id] = time.time()

    def _dispatch_all(self, prediction):
        emoji = RISK_EMOJI.get(prediction.risk_level, "⚠️")
        msg   = (
            f"{emoji} FLOOD ALERT — {prediction.risk_level}\n"
            f"📍 {prediction.location_name} ({prediction.camera_id})\n"
            f"💧 Depth: {prediction.water_depth_cm} cm\n"
            f"🎯 Confidence: {prediction.confidence_pct}%\n"
            f"🕐 {datetime.now().strftime('%d %b %Y %H:%M:%S')}\n"
            f"➡️  {prediction.recommended_action}"
        )
        if self.webhook_url:
            self._send_webhook(msg, prediction)
        for number in self.sms_numbers:
            self._send_sms(msg, number)

    def _send_webhook(self, msg: str, prediction):
        payload = {
            "text": msg,
            "attachments": [{
                "color":  "#e74c3c" if prediction.risk_level != "CRITICAL" else "#7b241c",
                "fields": [
                    {"title": "Camera",     "value": prediction.camera_id,        "short": True},
                    {"title": "Location",   "value": prediction.location_name,    "short": True},
                    {"title": "Depth",      "value": f"{prediction.water_depth_cm} cm", "short": True},
                    {"title": "Risk",       "value": prediction.risk_level,       "short": True},
                    {"title": "Confidence", "value": f"{prediction.confidence_pct}%", "short": True},
                    {"title": "Coords",
                     "value": f"{prediction.latitude}, {prediction.longitude}",   "short": True},
                ],
            }]
        }
        try:
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(
                self.webhook_url, data=data,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
            log.info("Webhook alert sent for %s", prediction.camera_id)
        except Exception:
            log.exception("Webhook alert failed for %s", prediction.camera_id)

    def _send_sms(self, msg: str, to_number: str):
        try:
            from twilio.rest import Client
            client = Client(self.twilio_sid, self.twilio_token)
            client.messages.create(body=msg[:1600], from_=self.twilio_from, to=to_number)
            log.info("SMS alert sent to %s for %s", to_number, msg[:40])
        except ImportError:
            log.warning("twilio not installed — SMS not sent. pip install twilio")
        except Exception:
            log.exception("SMS alert failed to %s", to_number)
