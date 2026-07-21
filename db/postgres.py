# db/postgres.py
"""
PostgreSQL Writer Module
=========================
Completely decoupled from the pipeline — receives FloodPrediction objects
and persists them.  The UI reads from the same DB independently.

Schema (auto-created on first run):
  flood_readings (
    id              SERIAL PRIMARY KEY,
    batch_id        TEXT,
    camera_id       TEXT,
    location_id     TEXT,
    location_name   TEXT,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    captured_at     TIMESTAMPTZ,
    flood_detected  BOOLEAN,
    water_depth_cm  REAL,
    risk_level      TEXT,
    recommended_action TEXT,
    confidence_pct  REAL,
    water_coverage_pct REAL,
    mean_flood_depth_cm REAL,
    max_flood_depth_cm  REAL,
    calibration_source  TEXT,
    seg_engine      TEXT,
    yolo_engine     TEXT,
    depth_engine    TEXT,
    inserted_at     TIMESTAMPTZ DEFAULT NOW()
  )

Usage:
    from db.postgres import PostgresWriter, DB_URL
    writer = PostgresWriter(DB_URL)
    writer.upsert(prediction)
"""

import logging
import os
from typing import List

log = logging.getLogger("db.postgres")

# Read from environment or .env file — never hardcode credentials
DB_URL = os.environ.get(
    "FLOODWATCH_DB_URL",
    "postgresql://floodwatch:floodwatch@localhost:5432/floodwatch"
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS flood_readings (
    id                  SERIAL PRIMARY KEY,
    batch_id            TEXT        NOT NULL,
    camera_id           TEXT        NOT NULL,
    location_id         TEXT,
    location_name       TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    captured_at         TIMESTAMPTZ,
    flood_detected      BOOLEAN,
    water_depth_cm      REAL,
    risk_level          TEXT,
    recommended_action  TEXT,
    confidence_pct      REAL,
    water_coverage_pct  REAL,
    mean_flood_depth_cm REAL,
    max_flood_depth_cm  REAL,
    calibration_source  TEXT,
    seg_engine          TEXT,
    yolo_engine         TEXT,
    depth_engine        TEXT,
    inserted_at         TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_flood_camera    ON flood_readings(camera_id);
CREATE INDEX IF NOT EXISTS idx_flood_batch     ON flood_readings(batch_id);
CREATE INDEX IF NOT EXISTS idx_flood_inserted  ON flood_readings(inserted_at DESC);
"""

_UPSERT = """
INSERT INTO flood_readings (
    batch_id, camera_id, location_id, location_name,
    latitude, longitude, captured_at,
    flood_detected, water_depth_cm, risk_level, recommended_action,
    confidence_pct, water_coverage_pct, mean_flood_depth_cm, max_flood_depth_cm,
    calibration_source, seg_engine, yolo_engine, depth_engine
) VALUES (
    %(batch_id)s, %(camera_id)s, %(location_id)s, %(location_name)s,
    %(latitude)s, %(longitude)s, %(captured_at)s,
    %(flood_detected)s, %(water_depth_cm)s, %(risk_level)s, %(recommended_action)s,
    %(confidence_pct)s, %(water_coverage_pct)s, %(mean_flood_depth_cm)s, %(max_flood_depth_cm)s,
    %(calibration_source)s, %(seg_engine)s, %(yolo_engine)s, %(depth_engine)s
)
ON CONFLICT DO NOTHING;
"""


class PostgresWriter:
    """Thread-safe PostgreSQL writer. Creates table on first connect."""

    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self._conn  = None
        self._connect()

    def _connect(self):
        try:
            import psycopg2
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = False
            with self._conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
            self._conn.commit()
            log.info("PostgreSQL connected and schema ready")
        except Exception:
            log.warning("PostgreSQL unavailable — writes will be skipped", exc_info=True)
            self._conn = None

    def upsert(self, prediction) -> bool:
        """Write one FloodPrediction row. Returns True on success."""
        if self._conn is None:
            log.debug("DB not connected — skipping write for %s", prediction.camera_id)
            return False
        try:
            row = {
                "batch_id":            prediction.batch_id,
                "camera_id":           prediction.camera_id,
                "location_id":         prediction.location_id,
                "location_name":       prediction.location_name,
                "latitude":            prediction.latitude,
                "longitude":           prediction.longitude,
                "captured_at":         prediction.timestamp,
                "flood_detected":      prediction.flood_detected,
                "water_depth_cm":      prediction.water_depth_cm,
                "risk_level":          prediction.risk_level,
                "recommended_action":  prediction.recommended_action,
                "confidence_pct":      prediction.confidence_pct,
                "water_coverage_pct":  prediction.water_coverage_pct,
                "mean_flood_depth_cm": prediction.mean_flood_depth_cm,
                "max_flood_depth_cm":  prediction.max_flood_depth_cm,
                "calibration_source":  prediction.calibration_source,
                "seg_engine":          prediction.seg_engine,
                "yolo_engine":         prediction.yolo_engine,
                "depth_engine":        prediction.depth_engine,
            }
            with self._conn.cursor() as cur:
                cur.execute(_UPSERT, row)
            self._conn.commit()
            return True
        except Exception:
            self._conn.rollback()
            log.exception("DB write failed for %s", prediction.camera_id)
            return False

    def upsert_batch(self, predictions: List) -> int:
        """Write multiple predictions. Returns count of successful writes."""
        return sum(1 for p in predictions if self.upsert(p))

    def latest_per_camera(self) -> List[dict]:
        """Return the most recent reading per camera — used by the UI."""
        if self._conn is None:
            return []
        sql = """
        SELECT DISTINCT ON (camera_id)
            camera_id, location_id, location_name, latitude, longitude,
            flood_detected, water_depth_cm, risk_level, recommended_action,
            confidence_pct, water_coverage_pct, captured_at, batch_id,
            calibration_source, seg_engine, yolo_engine, depth_engine
        FROM flood_readings
        ORDER BY camera_id, inserted_at DESC;
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            log.exception("DB read failed")
            return []

    def close(self):
        if self._conn:
            self._conn.close()
