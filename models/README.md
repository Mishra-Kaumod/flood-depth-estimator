# Flood Depth Model Storage

Pre-trained model is now included in the repository via **Git LFS** for easy team access.

## Available Models

| File | Size | Architecture | Status |
|------|------|-------------|--------|
| `best_floodnet_v2.pth` | 20.6 MB | EfficientNet-B4, multi-task | ✅ **Production (current)** |
| `best_flood_model_water_aware.pth` | 52.9 MB | EfficientNet-B0, water-aware loss | Previous production |
| `best_flood_model.pth` | 44.8 MB | EfficientNet-B0, MSE loss | Baseline |

Active model is set in `config/config.yaml` → `inference.model_path`.

---

## Quick Start

### Model Available in Repository
- **File:** `models/best_floodnet_v2.pth` ← current production
- **Size:** 42.72 MB (stored via Git LFS)
- **Architecture:** EfficientNet-B0
- **Input:** 224x224 RGB image
- **Output:** Severity classification + depth in cm (0-100)
- **Framework:** PyTorch

### Using the Pre-trained Model

```python
from src.train import build_model
import torch

# Load pre-trained model
model = build_model()
model.load_state_dict(torch.load('models/best_flood_model.pth'))
model.eval()

# Inference
from PIL import Image
image = Image.open('flood_image.jpg')
# ... preprocess and predict
```

### Running the Inference Server

```bash
# Start LitServe inference server
python serve.py

# Server runs on http://localhost:8000
# Automatically loads models/best_flood_model.pth
```

## For Development (Local Training)

When training locally, new models save to:
```bash
python src/train.py --config config/config.yaml --output models
# Creates: models/best_flood_model.pth (overwrites if exists)
```

To preserve trained model:
```bash
cp models/best_flood_model.pth models/best_flood_model_v2.pth
```

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
| Architecture | EfficientNet-B0 |
| Parameters | 4.37 million |
| Input Size | 224x224 RGB |
| Output | Depth (0-100 cm) |
| Framework | PyTorch |
| Training Loss | MSE |
| Optimizer | AdamW |
| Trained On | Flood Depth Dataset |

## Distribution Methods

### Method 1: Clone from GitHub (CURRENT - Recommended for teams <20 people)
```bash
git clone https://github.com/Mishra-Kaumod/flood-depth-estimator.git
# Model included via Git LFS
```

### Method 2: Direct Download Link
```bash
wget https://github.com/Mishra-Kaumod/flood-depth-estimator/releases/download/v1.0/best_flood_model.pth
# Or download via GitHub Releases page
```

### Method 3: AWS S3 (Production)
```bash
aws s3 cp s3://flood-depth-models/v1/best_flood_model.pth models/
```

## Best Practices

1. ✅ Always use Git LFS for >10 MB files
2. ✅ Commit `.gitattributes` with LFS configuration
3. ✅ Keep production model updated with latest version
4. ✅ Version models: `best_flood_model_v1.pth`, `v2.pth`, etc.
5. ✅ Document model training date and dataset version
6. ✅ Monitor Git LFS bandwidth usage (GitHub dashboard)

## Troubleshooting

### Model Not Downloaded After Clone
```bash
git lfs pull  # Manually pull LFS objects
git lfs fsck  # Verify integrity
```

### File Shows as Pointer Instead of Model
```bash
git lfs install  # Reinstall Git LFS
git lfs pull     # Pull actual file
```

### Upload Large Model
```bash
git lfs track models/best_flood_model.pth
git add models/best_flood_model.pth .gitattributes
git commit -m "Update: pre-trained model"
git push origin main
```

