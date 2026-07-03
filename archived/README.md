# flood-depth-estimator

## Model weights integration

Prediction scripts now resolve model files from either environment variables or repository paths.

- `FLOOD_MODEL_PATH` for flood/depth model weights
- `SEVERITY_MODEL_PATH` for severity classifier weights

If env vars are not set, scripts auto-search these repo locations:

- Flood model: `models/flood_model_final.pth`, `flood_model_final.pth`, `depth_classifier.pth`
- Severity model: `models/severity_efficientnet.pth`, `models/severity_model.pth`, `severity_efficientnet.pth`, `severity_model.pth`

## Bengaluru dashboard architecture (AWS-ready modular layout)

Dashboard mapping logic is split into reusable services for easier deployment and scaling:

- `flood_api/services/location_mapping.py`: camera location resolution and randomized Bengaluru fallback mapping.
- `flood_api/services/map_payload.py`: telemetry-to-map-point payload conversion and 5-point depth intensity scale.

When lat/lon are missing, uploads are auto-mapped to deterministic randomized Bengaluru locations.

Optional configuration for cloud environments:

- `BENGALURU_CAMERA_POINTS_JSON`: JSON array of `{name, latitude, longitude}` objects to override default map points.

## Enterprise API surface (v1)

- `GET /api/v1/health/live/` - liveness check
- `GET /api/v1/health/ready/` - readiness check (database + redis)
- `POST /api/v1/ingest/batch/` - batch upload with `images` + optional `camera_id`, `location_id`, `latitude`, `longitude`
- `GET /api/v1/telemetry/map-points/` - Bengaluru map points for web/mobile clients
- `GET /api/v1/telemetry/<telemetry_id>/` - telemetry JSON detail
- `POST /api/v1/feedback/` - reviewer feedback feedloop (`accepted/rejected/corrected`)
- `GET /api/v1/feedback/queue/` - low-confidence telemetry queue
- `GET|POST /api/v1/models/` - model version registry metadata

## Using E drive for runtime data

The app now supports runtime offloading to E drive:

- Default runtime root: `E:\flood_runtime` (when E drive exists)
- Upload temp files: `E:\flood_runtime\tmp_uploads`
- Model/cache directories (Torch/HuggingFace/Ultralytics): under `E:\flood_runtime`

Override with:

- `FLOOD_RUNTIME_ROOT=<your_path>`

## Android app scaffold

An Android starter app is included at `android_app/` (Jetpack Compose + Google Maps + backend API client):
- Renders map markers from `/api/v1/telemetry/map-points/`
- Supports backend-connected architecture for production mobile rollout