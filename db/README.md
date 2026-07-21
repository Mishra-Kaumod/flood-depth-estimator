# db/ — PostgreSQL Database Layer

Handles all reads and writes to the predictions database.

## Tables

| Table | Purpose |
|-------|---------|
| `predictions` | Every image prediction result (depth, risk, timestamp, location) |
| `training_samples` | Labelled images for MLOps retraining |
| `model_registry` | Model versions, accuracy scores, promotion history |

## Usage
```python
from db.postgres import DBWriter

writer = DBWriter()
writer.write_prediction({
    "filename": "flood_001.jpg",
    "flood_detected": True,
    "water_depth_cm": 45,
    "risk_level": "moderate",
    "water_coverage_pct": 62,
})
```

## Connection
Set `DATABASE_URL` in `.env`:
```
DATABASE_URL=postgresql://user:password@localhost:5432/floodwatch
```
