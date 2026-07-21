# pipeline/ — 7-Stage Inference Pipeline

Each file is one stage. `runner.py` wires them in order.

| Stage | File | Input → Output |
|-------|------|----------------|
| 1 | `segformer.py` | Raw image → water mask (binary) |
| 2 | `yolo.py` | Raw image → reference objects (bounding boxes) |
| 3 | `depth.py` | Raw image → dense depth map |
| 4 | `fusion.py` | mask + depth + objects → calibrated depth (cm) |
| 5 | `severity.py` | calibrated depth → risk level (5-class) |
| 6 | `gemini_validator.py` | *(optional)* Gemini API ensemble confirmation |
| `runner.py` | Orchestrates stages 1–6 end-to-end |

## Usage
```python
from pipeline.runner import run_pipeline
result = run_pipeline("path/to/flood_image.jpg")
# result = { flood_detected, water_depth_cm, risk_level, water_coverage_pct }
```

## Adding a new stage
1. Create `pipeline/my_stage.py` with a `run(image, context) -> context` function
2. Import and call it in `runner.py` between existing stages
