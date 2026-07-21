# api/server.py
"""
FloodWatch AI — FastAPI REST Server
=====================================
Exposes the full pipeline as HTTP endpoints.
Use this when cameras POST images directly, or for integration testing.

Endpoints:
  POST /predict          — single image with metadata → 5 outputs + Gemini
  POST /predict/batch    — multiple images in one request
  GET  /health           — liveness check
  GET  /readings/latest  — latest per-camera from PostgreSQL

Run:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Test:
  curl -X POST http://localhost:8000/predict \
       -F "image=@flood.jpg" \
       -F "camera_id=CAM_001" \
       -F "location_id=LOC_001" \
       -F "latitude=12.9172" \
       -F "longitude=77.6228" \
       -F "location_name=SilkBoard"
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi            import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses  import JSONResponse
from pydantic           import BaseModel
import cv2
import numpy as np

from pipeline.runner    import PipelineRunner
from ingestor           import CameraImage
from db.postgres        import PostgresWriter, DB_URL

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "FloodWatch AI API",
    description = "Flood depth estimation — SegFormer + YOLOv8 + Depth Anything V2 + Gemini",
    version     = "3.0.0",
)

_pipeline: Optional[PipelineRunner] = None
_writer:   Optional[PostgresWriter] = None


def get_pipeline() -> PipelineRunner:
    global _pipeline
    if _pipeline is None:
        cfg = {"pipeline": {
            "device":         "cpu",
            "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
        }}
        _pipeline = PipelineRunner(cfg)
    return _pipeline


def get_writer() -> PostgresWriter:
    global _writer
    if _writer is None:
        _writer = PostgresWriter(DB_URL)
    return _writer


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    flood_detected:      bool
    water_depth_cm:      float
    risk_level:          str
    recommended_action:  str
    confidence_pct:      float
    camera_id:           str
    location_id:         str
    latitude:            float
    longitude:           float
    location_name:       str
    timestamp:           str
    ensemble_method:     str
    gemini_risk:         Optional[str]   = None
    gemini_depth_cm:     Optional[float] = None
    gemini_confidence:   Optional[float] = None
    gemini_reasoning:    Optional[str]   = None
    gemini_agreement:    Optional[bool]  = None


class HealthResponse(BaseModel):
    status:         str
    pipeline_ready: bool
    gemini_enabled: bool
    db_connected:   bool
    timestamp:      str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    pipeline = get_pipeline()
    writer   = get_writer()
    return HealthResponse(
        status         = "ok",
        pipeline_ready = pipeline is not None,
        gemini_enabled = pipeline.gemini.enabled if pipeline else False,
        db_connected   = writer._conn is not None,
        timestamp      = datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    image:         UploadFile = File(...,  description="Camera image (JPG/PNG)"),
    camera_id:     str        = Form(...,  example="CAM_001"),
    location_id:   str        = Form(...,  example="LOC_001"),
    latitude:      float      = Form(...,  example=12.9172),
    longitude:     float      = Form(...,  example=77.6228),
    location_name: str        = Form("",   example="Silk Board Junction"),
):
    """
    Run the full 5-stage pipeline on a single image.
    Gemini ensemble runs automatically if GEMINI_API_KEY is set.
    Result is persisted to PostgreSQL.
    """
    raw     = await image.read()
    arr     = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Cannot decode image.")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img_bgr)
        tmp_path = Path(f.name)

    try:
        cam_img = CameraImage(
            image_path    = tmp_path,
            camera_id     = camera_id,
            location_id   = location_id,
            latitude      = latitude,
            longitude     = longitude,
            location_name = location_name,
            captured_at   = datetime.now().isoformat(),
        )
        pred = get_pipeline().run_image(cam_img, batch_id="api")
    finally:
        tmp_path.unlink(missing_ok=True)

    try:
        get_writer().upsert(pred)
    except Exception:
        pass

    return PredictResponse(
        flood_detected     = pred.flood_detected,
        water_depth_cm     = pred.water_depth_cm,
        risk_level         = pred.risk_level,
        recommended_action = pred.recommended_action,
        confidence_pct     = pred.confidence_pct,
        camera_id          = pred.camera_id,
        location_id        = pred.location_id,
        latitude           = pred.latitude,
        longitude          = pred.longitude,
        location_name      = pred.location_name,
        timestamp          = pred.timestamp,
        ensemble_method    = getattr(pred, "ensemble_method", "model_only"),
        gemini_risk        = getattr(pred, "gemini_risk",       None),
        gemini_depth_cm    = getattr(pred, "gemini_depth_cm",   None),
        gemini_confidence  = getattr(pred, "gemini_confidence", None),
        gemini_reasoning   = getattr(pred, "gemini_reasoning",  None),
        gemini_agreement   = getattr(pred, "gemini_agreement",  None),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    images:        List[UploadFile] = File(...),
    camera_id:     str              = Form(...),
    location_id:   str              = Form(...),
    latitude:      float            = Form(...),
    longitude:     float            = Form(...),
    location_name: str              = Form(""),
):
    """Run prediction on multiple images from the same camera location."""
    results = []
    for i, img_file in enumerate(images):
        raw     = await img_file.read()
        arr     = np.frombuffer(raw, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            results.append({"file": img_file.filename, "error": "Could not decode"})
            continue

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img_bgr)
            tmp_path = Path(f.name)

        try:
            cam_img = CameraImage(
                image_path    = tmp_path,
                camera_id     = f"{camera_id}_{i+1:02d}",
                location_id   = location_id,
                latitude      = latitude,
                longitude     = longitude,
                location_name = location_name,
                captured_at   = datetime.now().isoformat(),
            )
            pred = get_pipeline().run_image(cam_img, batch_id="api_batch")
            get_writer().upsert(pred)
            results.append({
                "file":            img_file.filename,
                "flood_detected":  pred.flood_detected,
                "water_depth_cm":  pred.water_depth_cm,
                "risk_level":      pred.risk_level,
                "confidence_pct":  pred.confidence_pct,
                "ensemble_method": getattr(pred, "ensemble_method", "model_only"),
            })
        except Exception as e:
            results.append({"file": img_file.filename, "error": str(e)})
        finally:
            tmp_path.unlink(missing_ok=True)

    return {"batch_size": len(images), "results": results}


@app.get("/readings/latest", tags=["Data"])
def latest_readings():
    """Return latest prediction per camera from PostgreSQL."""
    return get_writer().latest_per_camera()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
