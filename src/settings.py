"""
Typed configuration loader with environment overlays and secret resolution.

This module validates config/config.yaml at startup using Pydantic models,
merges environment-specific overrides, and resolves secret placeholders.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field, ValidationError

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # pragma: no cover
    BaseSettings = BaseModel  # type: ignore[misc,assignment]

    class SettingsConfigDict(dict):  # type: ignore[override]
        pass


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLOOD_", extra="ignore")

    app_env: str = "production"
    config_path: str = "config/config.yaml"


class AggregatorThresholds(BaseModel):
    advisory_m: float = 0.10
    warning_m: float = 0.30
    alert_m: float = 0.60
    critical_m: float = 1.00
    advisory_cm: float = 10.0
    warning_cm: float = 30.0
    alert_cm: float = 60.0
    critical_cm: float = 100.0


class AggregatorConfig(BaseModel):
    window_seconds: int = Field(default=600, ge=10)
    burst_threshold: int = Field(default=5, ge=1)
    backend: str = "memory"
    redis_url: str = "redis://localhost:6379/1"
    thresholds: AggregatorThresholds = Field(default_factory=AggregatorThresholds)


class AppConfig(BaseModel):
    training: Dict[str, Any]
    data: Dict[str, Any]
    inference: Dict[str, Any]
    event_processing: Dict[str, Any] = Field(default_factory=dict)
    aws: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, Any] = Field(default_factory=dict)
    environments: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    aggregator: AggregatorConfig = Field(default_factory=AggregatorConfig)
    secrets: Dict[str, str] = Field(default_factory=dict)


_SECRET_PATTERN = re.compile(r"^\$\{SECRET:([A-Z0-9_]+)\}$")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _set_deep(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _apply_environment_overrides(config: Dict[str, Any], env_name: str) -> Dict[str, Any]:
    overrides = config.get("environments", {}).get(env_name, {})
    if not isinstance(overrides, dict):
        return config

    merged = copy.deepcopy(config)
    for key, value in overrides.items():
        if "." in key:
            _set_deep(merged, key, value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _resolve_secret_placeholders(config: Dict[str, Any]) -> Dict[str, Any]:
    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            m = _SECRET_PATTERN.match(value.strip())
            if m:
                env_key = m.group(1)
                env_val = os.getenv(env_key)
                if env_val is None:
                    raise ValueError(f"Missing required secret environment variable: {env_key}")
                return env_val
        return value

    return walk(config)


def load_settings(config_path: str | None = None, env_name: str | None = None) -> AppConfig:
    runtime = RuntimeSettings()
    path = Path(config_path or runtime.config_path)
    environment = env_name or runtime.app_env

    raw = _load_yaml(path)
    overlaid = _apply_environment_overrides(raw, environment)
    resolved = _resolve_secret_placeholders(overlaid)

    try:
        return AppConfig.model_validate(resolved)
    except ValidationError as exc:  # pragma: no cover
        raise ValueError(f"Configuration validation failed: {exc}") from exc


def load_settings_dict(config_path: str | None = None, env_name: str | None = None) -> Dict[str, Any]:
    return load_settings(config_path=config_path, env_name=env_name).model_dump()
