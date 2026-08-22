"""
SQLite persistence for camera locations, telemetry, and temporal sequences.

This is the cleaned-project equivalent of the old Django models in
wellabs latest. It keeps the same domain records without taking on Django.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from src.settings import load_settings_dict


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else [])


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


@dataclass
class CameraLocationRecord:
    camera_id: str
    location_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: str = ""
    created_at: str = ""


@dataclass
class TelemetryRecord:
    id: Optional[int]
    timestamp: str
    image_name: str
    camera_id: str
    strategy_applied: str
    surface_water_confirmed_pct: float
    computed_depth_cm: float
    system_confidence_score_pct: float
    detected_reference_objects: list[str]
    num_reference_objects: int
    is_water_confirmed: bool
    safety_risk_assessment: str
    metadata: dict[str, Any]


@dataclass
class TemporalSequenceRecord:
    id: Optional[int]
    camera_id: str
    sequence_start: str
    sequence_end: str
    image_count: int
    average_depth_cm: Optional[float]
    max_depth_cm: Optional[float]
    min_depth_cm: Optional[float]
    water_detected_in_images: int
    detected_anchor_types: list[str]
    consensus_water_present: bool
    confidence_score: float
    telemetry_record_ids: list[int]
    created_at: str


class FloodRepository:
    def __init__(self, db_path: str | None = None, config_path: str = "config/config.yaml"):
        cfg = load_settings_dict(config_path=config_path)
        storage_cfg = cfg.get("storage", {})
        self.db_path = Path(db_path or storage_cfg.get("sqlite_path", "data/flood_app.sqlite3"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS camera_locations (
                    camera_id TEXT PRIMARY KEY,
                    location_name TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS flood_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_name TEXT,
                    camera_id TEXT,
                    strategy_applied TEXT NOT NULL,
                    surface_water_confirmed_pct REAL NOT NULL,
                    computed_depth_cm REAL NOT NULL,
                    system_confidence_score_pct REAL NOT NULL,
                    detected_reference_objects TEXT NOT NULL,
                    num_reference_objects INTEGER NOT NULL,
                    is_water_confirmed INTEGER NOT NULL,
                    safety_risk_assessment TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY(camera_id) REFERENCES camera_locations(camera_id)
                );

                CREATE INDEX IF NOT EXISTS idx_telemetry_camera_timestamp
                    ON flood_telemetry(camera_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_telemetry_confirmed_timestamp
                    ON flood_telemetry(is_water_confirmed, timestamp DESC);

                CREATE TABLE IF NOT EXISTS temporal_sequences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    sequence_start TEXT NOT NULL,
                    sequence_end TEXT NOT NULL,
                    image_count INTEGER NOT NULL,
                    average_depth_cm REAL,
                    max_depth_cm REAL,
                    min_depth_cm REAL,
                    water_detected_in_images INTEGER NOT NULL,
                    detected_anchor_types TEXT NOT NULL,
                    consensus_water_present INTEGER NOT NULL,
                    confidence_score REAL NOT NULL,
                    telemetry_record_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(camera_id) REFERENCES camera_locations(camera_id)
                );

                CREATE INDEX IF NOT EXISTS idx_sequences_camera_start
                    ON temporal_sequences(camera_id, sequence_start DESC);
                """
            )

    def upsert_camera(
        self,
        camera_id: str,
        location_name: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        description: str = "",
    ) -> CameraLocationRecord:
        existing = self.get_camera(camera_id)
        if existing is None:
            record = CameraLocationRecord(
                camera_id=camera_id,
                location_name=location_name or f"Location {camera_id}",
                latitude=latitude,
                longitude=longitude,
                description=description,
                created_at=_utc_now(),
            )
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO camera_locations
                    (camera_id, location_name, latitude, longitude, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.camera_id,
                        record.location_name,
                        record.latitude,
                        record.longitude,
                        record.description,
                        record.created_at,
                    ),
                )
            return record

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE camera_locations
                SET location_name = COALESCE(?, location_name),
                    latitude = COALESCE(?, latitude),
                    longitude = COALESCE(?, longitude),
                    description = CASE WHEN ? != '' THEN ? ELSE description END
                WHERE camera_id = ?
                """,
                (location_name, latitude, longitude, description, description, camera_id),
            )
        return self.get_camera(camera_id) or existing

    def get_camera(self, camera_id: str) -> CameraLocationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM camera_locations WHERE camera_id = ?", (camera_id,)
            ).fetchone()
        if row is None:
            return None
        return CameraLocationRecord(**dict(row))

    def list_cameras(self) -> list[CameraLocationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM camera_locations ORDER BY camera_id"
            ).fetchall()
        return [CameraLocationRecord(**dict(row)) for row in rows]

    def save_telemetry(self, record: TelemetryRecord) -> TelemetryRecord:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO flood_telemetry
                (timestamp, image_name, camera_id, strategy_applied,
                 surface_water_confirmed_pct, computed_depth_cm,
                 system_confidence_score_pct, detected_reference_objects,
                 num_reference_objects, is_water_confirmed, safety_risk_assessment,
                 metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp or _utc_now(),
                    record.image_name,
                    record.camera_id,
                    record.strategy_applied,
                    record.surface_water_confirmed_pct,
                    record.computed_depth_cm,
                    record.system_confidence_score_pct,
                    _json(record.detected_reference_objects),
                    record.num_reference_objects,
                    1 if record.is_water_confirmed else 0,
                    record.safety_risk_assessment,
                    _json(record.metadata),
                ),
            )
            record.id = int(cur.lastrowid)
        return record

    def recent_telemetry(self, limit: int = 20, camera_id: str | None = None) -> list[TelemetryRecord]:
        query = "SELECT * FROM flood_telemetry"
        params: list[Any] = []
        if camera_id:
            query += " WHERE camera_id = ?"
            params.append(camera_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._telemetry_from_row(row) for row in rows]

    def telemetry_since(self, camera_id: str, since_iso: str) -> list[TelemetryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM flood_telemetry
                WHERE camera_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (camera_id, since_iso),
            ).fetchall()
        return [self._telemetry_from_row(row) for row in rows]

    def save_temporal_sequence(self, record: TemporalSequenceRecord) -> TemporalSequenceRecord:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO temporal_sequences
                (camera_id, sequence_start, sequence_end, image_count,
                 average_depth_cm, max_depth_cm, min_depth_cm,
                 water_detected_in_images, detected_anchor_types,
                 consensus_water_present, confidence_score,
                 telemetry_record_ids, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.camera_id,
                    record.sequence_start,
                    record.sequence_end,
                    record.image_count,
                    record.average_depth_cm,
                    record.max_depth_cm,
                    record.min_depth_cm,
                    record.water_detected_in_images,
                    _json(record.detected_anchor_types),
                    1 if record.consensus_water_present else 0,
                    record.confidence_score,
                    _json(record.telemetry_record_ids),
                    record.created_at or _utc_now(),
                ),
            )
            record.id = int(cur.lastrowid)
        return record

    def latest_temporal_sequence(self, camera_id: str) -> TemporalSequenceRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM temporal_sequences
                WHERE camera_id = ?
                ORDER BY sequence_start DESC
                LIMIT 1
                """,
                (camera_id,),
            ).fetchone()
        if row is None:
            return None
        return self._sequence_from_row(row)

    def camera_stats(self, camera_id: str, since_iso: str) -> dict[str, Any]:
        records = self.telemetry_since(camera_id, since_iso)
        depths = [r.computed_depth_cm for r in records]
        water_confirmed = sum(1 for r in records if r.is_water_confirmed)
        with self._connect() as conn:
            sequence_count = conn.execute(
                """
                SELECT COUNT(*) AS count FROM temporal_sequences
                WHERE camera_id = ? AND sequence_start >= ?
                """,
                (camera_id, since_iso),
            ).fetchone()["count"]
        return {
            "camera_id": camera_id,
            "total_images": len(records),
            "water_confirmed_images": water_confirmed,
            "avg_depth_cm": round(sum(depths) / len(depths), 2) if depths else 0,
            "max_depth_cm": max(depths) if depths else 0,
            "temporal_sequences": int(sequence_count),
        }

    def _telemetry_from_row(self, row: sqlite3.Row) -> TelemetryRecord:
        data = dict(row)
        data["detected_reference_objects"] = _loads(data["detected_reference_objects"], [])
        data["metadata"] = _loads(data["metadata"], {})
        data["is_water_confirmed"] = bool(data["is_water_confirmed"])
        return TelemetryRecord(**data)

    def _sequence_from_row(self, row: sqlite3.Row) -> TemporalSequenceRecord:
        data = dict(row)
        data["detected_anchor_types"] = _loads(data["detected_anchor_types"], [])
        data["telemetry_record_ids"] = _loads(data["telemetry_record_ids"], [])
        data["consensus_water_present"] = bool(data["consensus_water_present"])
        return TemporalSequenceRecord(**data)


def records_to_dicts(records: Iterable[Any]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
