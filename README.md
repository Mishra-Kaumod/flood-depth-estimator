# Bengaluru Flood Depth Estimator

A production-grade flood depth estimation system for Bengaluru using a 5-stage ML pipeline.
Given a flood image with GPS coordinates, it returns depth in centimetres and a severity grade.

---

## Architecture

```
Image
  │
  ▼ Stage 1 — Water Detection       src/water_region_detector.py
  ▼ Stage 2 — Reference Objects     YOLOv8  (yolov8n.pt)
  ▼ Stage 3 — Dense Depth Map       Depth Anything V2 proxy
  ▼ Stage 4 — Fusion Engine         src/segformer_yolo_depthv2_pipeline.py
  ▼ Stage 5 — Calibration/Severity  Classical + optional Gemini evaluator (75% weight)
```

Stages 1–2 are always classical (no API calls).  
Stages 3–5 are optionally enhanced by Gemini when `GEMINI_API_KEY` is set.

**Severity buckets:** SAFE (<5 cm) · LOW (5–20 cm) · MEDIUM (20–50 cm) · HIGH (50–80 cm) · CRITICAL (>80 cm)

---

## Repository Structure

```
flood-depth-estimator/
│
├── app.py                    # Flask web server + full UI (main entry point)
├── Dockerfile                # Container definition
├── requirements.txt          # Core runtime dependencies
├── requirements-production.txt
├── requirements-local.txt
├── pyproject.toml
├── yolov8n.pt                # YOLOv8 nano weights (Stage 2)
│
├── src/                      # Core ML & pipeline code
│   ├── segformer_yolo_depthv2_pipeline.py  # 5-stage pipeline (heart of system)
│   ├── water_region_detector.py            # Stage 1: water pixel detection + validation
│   ├── reference_depth_estimator.py        # Physics-based scale anchor estimator
│   ├── aggregator.py                       # Sliding-window burst aggregator
│   ├── dlq.py                              # Dead-letter queue for failed events
│   ├── event_contract.py                   # Pydantic event schemas
│   ├── pipeline.py                         # Event-driven batch processor
│   ├── settings.py                         # Config loader
│   ├── dataset.py                          # PyTorch dataset for training
│   ├── train.py                            # EfficientNet-B0 training loop
│   ├── train_water_aware.py                # Water-aware training (loss on water pixels only)
│   ├── compute_stats.py                    # Dataset statistics utility
│   ├── geospatial_classifier.py            # GPS zone → severity mapping
│   ├── middleware/
│   │   ├── observability.py               # Structured logging + metrics
│   │   └── retry.py                       # Exponential backoff retry
│   └── flood_depth/
│       └── segmentation_engine.py         # Legacy segmentation prototype
│
├── models/                   # Model weights (Git LFS tracked)
│   ├── best_flood_model_water_aware.pth   # EfficientNet-B0 (default, water-aware)
│   └── best_flood_model.pth               # EfficientNet-B0 (base)
│
├── config/
│   └── config.yaml           # All runtime config (pipeline mode, thresholds, DLQ, aggregator)
│
├── scripts/                  # Standalone utilities (not imported by server)
│   ├── retrain_flood_classifier.py   # Local retraining script
│   ├── mc_dropout.py                 # Monte Carlo dropout confidence estimation
│   ├── serve.py                      # Production server entry (Gunicorn/waitress)
│   └── tasks.py                      # Background task definitions
│
├── notebooks/
│   └── Flood_Depth_Google_Colab.ipynb  # Colab training notebook (free GPU)
│
├── docs/                     # All documentation
│   ├── QUICKSTART.md                   # Local setup guide
│   ├── COLAB_RETRAINING_GUIDE.md       # Full Colab retraining walkthrough
│   ├── COLAB_QUICK_START.md            # Colab quick start
│   ├── GOOGLE_COLAB_GUIDE.md           # Colab environment setup
│   ├── INCREMENTAL_IMPROVEMENT_GUIDE.md # 5 improvement options without full retrain
│   ├── MODEL_DATASET_MANAGEMENT.md     # Model versioning and dataset management
│   ├── PRODUCTION_ARCHITECTURE.md      # Full production deployment architecture
│   ├── TRAINING_ROADMAP.md             # Planned ML improvements
│   ├── AFTER_TRAINING_CHECKLIST.md     # Post-training steps
│   ├── pipeline_architecture.html      # Canonical architecture and file-registry dashboard
│   └── dashboard.html                  # Ops monitoring dashboard
│
├── .github/workflows/
│   ├── deploy.yml                      # CI/CD: build → test → deploy on push to main
│   └── model-readiness-gate.yml        # Blocks deploy if model quality gate fails
│
└── archived/                 # Superseded prototypes and research files (not used in production)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py

# 3. Open the UI
http://127.0.0.1:5000
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLOOD_PIPELINE_MODE` | `segformer_yolov8_depthv2_fusion` | Which pipeline to run |
| `GEMINI_API_KEY` | _(none)_ | Optional — enables Gemini re-evaluation at stages 3–5 |
| `MODEL_PATH` | `models/best_flood_model_water_aware.pth` | Override ML model path |
| `PORT` | `5000` | Server port |

---

## API

### `POST /predict-batch`
Upload one or more flood images with GPS coordinates.

**Form fields:**
- `images[]` — image files
- `lats[]` — latitude per image
- `lngs[]` — longitude per image
- `names[]` — area name per image

**Response:**
```json
{
  "results": [{
    "name": "Koramangala",
    "depth_cm": 55.7,
    "confidence": 0.72,
    "severity": { "level": "HIGH", "label": "High flood — avoid travel", "color": "#dc2626" },
    "method": "segformer_yolov8_depthv2_fusion",
    "no_reference_warning": false,
    "water_coverage": 0.52,
    "action_trigger": "Activate Traffic Management",
    "pipeline_trace": [...]
  }]
}
```

### `GET /health`
Returns pipeline status, model load state, Gemini availability.

---

## Incremental Model Improvement

No full retrain required. See `docs/INCREMENTAL_IMPROVEMENT_GUIDE.md` for all options.

| Option | Time | GPU | Improvement |
|--------|------|-----|-------------|
| Test-Time Augmentation | 0 min | No | 3–8% |
| Head fine-tune | 30 min | No | 5–15% |
| Progressive fine-tune | 1–2 hr | Yes | 10–25% |
| Water-aware fine-tune ✅ | 2–3 hr | Yes | 20–40% |
| Full retrain (Colab) | 4–6 hr | Colab | 50–60% |

---

## Key Design Decisions

- **Brown water** — detector tuned for muddy Bengaluru floods, not just clear blue water
- **No reference object = estimate, not refusal** — pipeline continues with SegFormer+DepthV2 only, confidence capped at 55%
- **Water signal gates depth** — if the "water" detected is textured like a road, depth is withheld entirely
- **Gemini is post-hoc** — classical pipeline always runs first; Gemini re-checks stages 3–5 only

---

## License
MIT