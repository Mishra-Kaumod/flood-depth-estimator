import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from web_app import create_app
from archive.legacy_cli.modules.production_pipeline import ProductionFloodAnalyzer


def test_create_app_has_expected_routes():
    app = create_app()
    routes = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/" in routes
    assert "/health" in routes
    assert "/predict" in routes
    assert "/api/v1/estimate" in routes
    assert "/api/v1/temporal/<camera_id>" in routes
    assert "/api/v1/temporal/<camera_id>/analyze" in routes
    assert "/api/v1/camera/<camera_id>/stats" in routes
    assert "/api/v1/telemetry" in routes


def test_home_page_contains_upload_prompt():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Select an image" in body


def test_production_pipeline_returns_structured_result():
    analyzer = ProductionFloodAnalyzer()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = analyzer.analyze_bgr(image, "dummy.jpg")
    assert "image_path" in result
    assert "water_detected" in result
    assert "final_flood_level" in result
