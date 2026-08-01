# Flood Depth Model Storage

Pre-trained model is included in the repository via **Git LFS** for easy team access.

> **Only one model is active.** The old EfficientNet-B0 baseline has been moved to `archived/`.

## Quick Start

### Active Model
- **File:** `models/best_flood_model_water_aware.pth`
- **Architecture:** EfficientNet-B3 + water-aware head
- **Input:** 224×224 RGB image
- **Output:** Severity classification + depth in cm (0–300)
- **Framework:** PyTorch
- **Note:** Used by the full 5-stage pipeline (SegFormer → YOLOv8 → DepthV2 → Fusion → Calibration)

### Using the Active Model

```python
from src.train_water_aware import build_water_aware_model
import torch

model = build_water_aware_model()
model.load_state_dict(torch.load('models/best_flood_model_water_aware.pth'))
model.eval()
```

### Running the Inference Server

```bash
python app.py
# Server runs on http://localhost:5000
# Automatically loads models/best_flood_model_water_aware.pth
```

## For Development (Local Training / Fine-tuning)

```bash
python src/train_water_aware.py --config config/config.yaml --output models
# Creates: models/best_flood_model_water_aware.pth (overwrites if exists)
```

## Archived Models

| File | Architecture | Status |
|------|-------------|--------|
| `archived/best_flood_model.pth` | EfficientNet-B0 (baseline) | Superseded |
| `archived/severity_model.pth` | Early severity classifier | Superseded |

## Git LFS Details

This repository uses **Git Large File Storage (Git LFS)** to handle large model files efficiently.

### What is Git LFS?
- Stores large files separately from Git history
- Repository stores lightweight **pointer files** instead of full model
- Full model downloads only on first `git clone`
- Saves bandwidth and storage space

### Git LFS Tracking
```bash
# View tracked files
git lfs ls-files

# Track new .pth files
git lfs track "*.pth"
git add .gitattributes
```

### Clone Repository with Model

```bash
# Clone includes Git LFS setup
git clone https://github.com/Mishra-Kaumod/flood-depth-estimator.git

# Model is automatically downloaded
cd flood-depth-estimator
python serve.py  # Model ready to use!
```

## GitHub Cost Notes

### Git LFS Bandwidth
- GitHub provides **1 GB/month free** LFS bandwidth
- Model size: 42.72 MB
- ~23 clones/month covered by free tier
- Paid tier: $5/month for 50 GB additional bandwidth

### For High-Volume Teams
If team exceeds free tier:
1. Use AWS S3 for primary model storage
2. Reference S3 download in initialization
3. Keep LFS for backup/version control

## Model Specifications

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B3 + water-aware head |
| Input Size | 224×224 RGB |
| Output | Depth (0–300 cm) + severity bucket |
| Framework | PyTorch |
| Optimizer | AdamW |

## Distribution (Git LFS)

```bash
# Clone — model downloads automatically
git clone https://github.com/Mishra-Kaumod/flood-depth-estimator.git

# If model shows as pointer, pull manually
git lfs pull
```

> GitHub free tier: 1 GB/month LFS bandwidth (~23 full clones/month).

## Troubleshooting

```bash
git lfs install   # reinstall LFS hooks
git lfs pull      # force-download model
git lfs fsck      # verify integrity
```

