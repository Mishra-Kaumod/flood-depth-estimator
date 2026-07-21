# api/ — FastAPI REST API

Exposes the pipeline over HTTP. Use for external integrations (BBMP dashboard, mobile apps, etc.).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Upload image → get flood prediction JSON |
| GET | `/health` | Liveness check |
| POST | `/mlops/retrain` | Trigger model retraining |
| GET | `/mlops/status` | Retraining job status |
| GET | `/mlops/models` | List all model versions |
| POST | `/mlops/promote/{version}` | Promote model version to production |
| POST | `/mlops/rollback` | Rollback to previous model |

## Run locally
```bash
uvicorn api.server:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

## Auth
All endpoints (except `/health`) require `Authorization: Bearer <token>` header.  
Set `API_SECRET_KEY` in `.env` to configure.
