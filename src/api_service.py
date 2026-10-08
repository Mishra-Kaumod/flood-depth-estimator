"""V6 camera prediction and retained historical-data services.

Uploads use shared V6 inference only. Existing camera/telemetry/temporal reads
and temporal analytics concern historical records; they do not own V6 depth.
The V5 telemetry schema and automatic aggregation are not used by new uploads.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import Any

from src.event_contract import FloodEvent

from src.pipeline import execute_event

from src.storage import FloodRepository, records_to_dicts
from src.temporal_analysis import TemporalFloodAnalyzer

class FloodApiService:
    def __init__(self, repository: FloodRepository | None = None):
        self.repository = repository or FloodRepository()
        self.temporal_analyzer = TemporalFloodAnalyzer(self.repository)

    def process_camera_upload(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        camera_id: str,
        latitude: float,
        longitude: float,
        location_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Current V6 prediction; legacy telemetry/aggregation is not applied."""
        event = FloodEvent(
            source="api", camera_id=camera_id, latitude=latitude, longitude=longitude,
            image_b64=base64.b64encode(image_bytes).decode("ascii"),
            metadata={**(metadata or {}), "filename": filename, "location_name": location_name},
        )
        prediction = execute_event(event).to_api_response()
        return {"status": prediction["status"], "camera_id": camera_id, "result": prediction}

    def latest_temporal_sequence(self, camera_id: str) -> dict[str, Any] | None:
        sequence = self.temporal_analyzer.latest_sequence(camera_id)
        if sequence is None:
            return None
        return sequence.__dict__

    def trigger_temporal_analysis(self, camera_id: str, time_window_minutes: int = 15):
        return self.temporal_analyzer.create_temporal_sequence(camera_id, time_window_minutes)

    def camera_stats(self, camera_id: str, hours: int = 24) -> dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        camera = self.repository.get_camera(camera_id)
        stats = self.repository.camera_stats(camera_id, since.isoformat())
        stats.update(
            {
                "camera_name": camera.location_name if camera else f"Location {camera_id}",
                "hours_analyzed": hours,
            }
        )
        return stats

    def recent_telemetry(self, limit: int = 20, camera_id: str | None = None):
        return records_to_dicts(self.repository.recent_telemetry(limit=limit, camera_id=camera_id))
