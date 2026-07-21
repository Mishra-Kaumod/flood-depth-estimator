# config/settings.py
"""
Single source of truth for ALL configuration.
Reads from environment variables first, then .env file, then defaults.
Pass the Settings object to every module — no module reads env vars directly.

Usage:
    from config.settings import get_settings
    cfg = get_settings()
    print(cfg.db_url)
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name:    str  = "FloodWatch AI"
    app_version: str  = "3.0.0"
    environment: str  = Field("development", pattern="^(development|staging|production)$")
    log_level:   str  = "INFO"

    # ── Directories ───────────────────────────────────────────────────────────
    base_dir:        Path = Path(".")
    inbox_dir:       Path = Path("inbox")
    processing_dir:  Path = Path("processing")
    archive_dir:     Path = Path("archive")

    # ── Batch ingestion ───────────────────────────────────────────────────────
    batch_interval_seconds: int = 900
    max_batch_size:         int = 500

    # ── Database ──────────────────────────────────────────────────────────────
    db_url: str = Field(
        "postgresql://floodwatch:floodwatch@localhost:5432/floodwatch",
        alias="FLOODWATCH_DB_URL",
    )
    db_pool_size:    int = 10
    db_max_overflow: int = 20

    # ── Redis (job queue) ─────────────────────────────────────────────────────
    redis_url:          str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    queue_name:         str = "floodwatch:jobs"
    dead_letter_queue:  str = "floodwatch:dead"
    job_max_retries:    int = 3
    job_retry_delay_s:  int = 30

    # ── Pipeline models ───────────────────────────────────────────────────────
    pipeline_device:         str           = "cpu"
    segformer_weights:       Optional[Path] = None
    yolo_weights:            Optional[Path] = None
    depth_weights:           Optional[Path] = None
    severity_weights:        Optional[Path] = None
    sensor_height_cm:        float          = 300.0
    yolo_conf_threshold:     float          = 0.4
    pipeline_workers:        int            = 4

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_api_key:  str  = Field("", alias="GEMINI_API_KEY")
    gemini_model:    str  = "gemini-1.5-flash"
    gemini_enabled:  bool = True

    # ── API server ────────────────────────────────────────────────────────────
    api_host:           str  = "0.0.0.0"
    api_port:           int  = 8000
    api_key:            str  = Field("", alias="FLOODWATCH_API_KEY")
    api_key_required:   bool = True
    api_rate_limit:     int  = 60
    cors_origins:       str  = "*"

    # ── Alerts ────────────────────────────────────────────────────────────────
    alert_enabled:       bool = False
    alert_webhook_url:   str  = Field("", alias="ALERT_WEBHOOK_URL")
    alert_min_risk:      str  = "HIGH RISK"
    twilio_account_sid:  str  = Field("", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token:   str  = Field("", alias="TWILIO_AUTH_TOKEN")
    twilio_from_number:  str  = Field("", alias="TWILIO_FROM_NUMBER")
    alert_sms_numbers:   str  = Field("", alias="ALERT_SMS_NUMBERS")

    # ── Observability ─────────────────────────────────────────────────────────
    metrics_enabled:  bool = True
    metrics_port:     int  = 9090
    sentry_dsn:       str  = Field("", alias="SENTRY_DSN")

    @field_validator("inbox_dir", "processing_dir", "archive_dir", mode="before")
    @classmethod
    def resolve_path(cls, v):
        return Path(v)

    @property
    def gemini_active(self) -> bool:
        return bool(self.gemini_api_key) and self.gemini_enabled

    @property
    def sms_numbers_list(self) -> list[str]:
        return [n.strip() for n in self.alert_sms_numbers.split(",") if n.strip()]

    def pipeline_cfg(self) -> dict:
        return {
            "device":              self.pipeline_device,
            "segformer_weights":   str(self.segformer_weights) if self.segformer_weights else None,
            "yolo_weights":        str(self.yolo_weights)      if self.yolo_weights      else None,
            "depth_weights":       str(self.depth_weights)     if self.depth_weights     else None,
            "severity_weights":    str(self.severity_weights)  if self.severity_weights  else None,
            "sensor_height_cm":    self.sensor_height_cm,
            "yolo_conf_threshold": self.yolo_conf_threshold,
            "gemini_api_key":      self.gemini_api_key if self.gemini_active else "",
            "gemini_model":        self.gemini_model,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — reads .env once at first call."""
    return Settings()
