"""
MODEL & DATASET MANAGEMENT GUIDE
=================================

After training, you have:
1. Training images (100-300 files)
2. Trained model checkpoint (50MB)
3. Labels (CSV)
4. Training history (metrics)

This guide shows the BEST way to organize & version them.
"""

# ═════════════════════════════════════════════════════════════════════════════
# OPTION 1: Store in GitHub (Simple, small datasets)
# ═════════════════════════════════════════════════════════════════════════════

"""
Use this if: <200 images, want everything version-controlled

Folder structure:
    flood-depth-estimator/
    ├── models/
    │   ├── best_flood_model_water_aware.pth          (current model)
    │   └── archive/
    │       ├── v1_initial_collapse.pth               (failed model)
    │       ├── v2_50_images_from_kaggle.pth
    │       └── v3_150_mixed_sources.pth              (best so far)
    │
    ├── datasets/
    │   └── training_v3/                              (versioned dataset)
    │       ├── train/
    │       │   ├── flood_001.jpg
    │       │   └── ... 127 more
    │       ├── val/
    │       │   ├── flood_128.jpg
    │       │   └── ... 22 more
    │       ├── labels.csv                            (CRITICAL!)
    │       ├── metadata.json                         (track details)
    │       └── README.md
    │
    └── model_registry.json                           (version tracker)

Setup:
"""

import json
import os
from pathlib import Path
from datetime import datetime

# model_registry.json - Track all your models
MODEL_REGISTRY = {
    "models": [
        {
            "name": "best_flood_model_water_aware.pth",
            "version": "v3",
            "date_trained": "2026-07-03",
            "dataset": "training_v3",
            "images_count": 150,
            "train_images": 127,
            "val_images": 23,
            "epochs": 30,
            "best_val_loss": 0.0342,
            "val_loss_is_collapsed": False,
            "accuracy_metrics": {
                "mae_cm": 5.2,
                "rmse_cm": 8.7,
            },
            "sources": [
                "50 auto-downloaded (Wikimedia Commons)",
                "100 user-uploaded (Bengaluru streets)",
            ],
            "labels_method": "Gemini Pro + CV fallback",
            "notes": "First trained model, ready for deployment",
            "git_commit": "commit-hash-here",
        }
    ]
}

# datasets/training_v3/metadata.json
DATASET_METADATA = {
    "version": "v3",
    "created": "2026-07-03T13:00:00+05:30",
    "total_images": 150,
    "train": 127,
    "val": 23,
    "depth_statistics": {
        "mean_cm": 42.3,
        "std_cm": 28.4,
        "min_cm": 0,
        "max_cm": 150,
        "median_cm": 38.5,
    },
    "image_sources": {
        "kaggle": 20,
        "wikimedia_commons": 30,
        "user_uploads_bengaluru": 100,
    },
    "labeling": {
        "method": "Gemini 1.5 Flash",
        "fallback": "reference_depth_estimator CV",
        "gemini_success_rate": 0.92,
        "cv_fallback_used": 12,
    },
    "label_distribution": {
        "zero_cm": 18,
        "1_20_cm": 32,
        "20_50_cm": 48,
        "50_100_cm": 38,
        "100plus_cm": 14,
    },
    "quality_checks": {
        "no_label_collapse": True,
        "all_images_rgb": True,
        "min_resolution_200x200": True,
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# OPTION 2: DVC (Data Version Control) - Best for large/many datasets
# ═════════════════════════════════════════════════════════════════════════════

"""
Use this if: >500 images, multiple experiments, need versioning

Install:
    pip install dvc dvc-gdrive  # or dvc-s3, dvc-azure, etc.

Setup (one time):
    cd flood-depth-estimator
    git init
    dvc init
    dvc remote add -d myremote /mnt/shared/flood-datasets
    # OR: dvc remote add -d myremote s3://my-bucket/flood-data

Workflow:
    # Add large dataset to DVC
    dvc add datasets/training_v3/
    git add datasets/training_v3.dvc .gitignore
    git commit -m "Add training dataset v3"
    dvc push  # Push to remote storage (not GitHub!)

    # Later, clone + get data:
    git clone <repo>
    dvc pull  # Restores all large files
"""

DVC_SETUP = """
# .dvc/config
[remote "myremote"]
    url = /mnt/shared/flood-datasets

[core]
    remote = myremote
    autostage = true

# .gitignore (auto-generated)
/datasets/training_v3
"""


# ═════════════════════════════════════════════════════════════════════════════
# OPTION 3: Hugging Face Hub - Best for sharing + public models
# ═════════════════════════════════════════════════════════════════════════════

"""
Use this if: Want to share model with others, use for research

Benefits:
- Free cloud storage
- Model cards (documentation)
- Easy inference API
- Community engagement

Setup:
    pip install huggingface-hub

Upload:
"""

def upload_to_huggingface():
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi()
    
    # Create repo
    repo_id = "Mishra-Kaumod/flood-depth-estimator"
    create_repo(repo_id, repo_type="model", exist_ok=True)
    
    # Upload model
    api.upload_file(
        path_or_fileobj="models/best_flood_model_water_aware.pth",
        path_in_repo="best_flood_model_water_aware.pth",
        repo_id=repo_id,
    )
    
    # Upload metadata
    api.upload_file(
        path_or_fileobj="model_registry.json",
        path_in_repo="model_registry.json",
        repo_id=repo_id,
    )
    
    print(f"✅ Uploaded to https://huggingface.co/{repo_id}")


# ═════════════════════════════════════════════════════════════════════════════
# RECOMMENDED WORKFLOW (GitHub + DVC + metadata)
# ═════════════════════════════════════════════════════════════════════════════

"""
BEST PRACTICE for YOUR project:

Step 1: After training in Colab
    - Download model.pth
    - Download training_v3/ folder (images + labels.csv)
    - Note training metrics

Step 2: On your local machine
    # Save model
    cp best_flood_model_water_aware.pth models/archive/v3_$(date +%Y%m%d).pth
    cp best_flood_model_water_aware.pth models/best_flood_model_water_aware.pth
    
    # Save dataset (use DVC if large)
    mkdir -p datasets/training_v3/{train,val}
    cp -r ~/colab_downloads/training_v3/train/* datasets/training_v3/train/
    cp -r ~/colab_downloads/training_v3/val/* datasets/training_v3/val/
    cp ~/colab_downloads/labels.csv datasets/training_v3/

Step 3: Track in model_registry.json
    {
        "version": "v3",
        "date": "2026-07-03",
        "best_val_loss": 0.0342,
        "images": 150,
        "commit": "abc123def",
    }

Step 4: Push to GitHub
    git add models/best_flood_model_water_aware.pth model_registry.json
    git commit -m "Deploy model v3: 150 images, val_loss=0.0342"
    git push origin main

Step 5: OPTIONAL - HuggingFace
    # Share with community
    huggingface-cli upload Mishra-Kaumod/flood-depth-estimator \
        models/best_flood_model_water_aware.pth --repo-type=model
"""


# ═════════════════════════════════════════════════════════════════════════════
# CREATE DATASET README (for reproducibility)
# ═════════════════════════════════════════════════════════════════════════════

DATASET_README = """# Training Dataset v3

## Overview
- **Total images:** 150 (127 train, 23 val)
- **Date:** 2026-07-03
- **Sources:** Kaggle (50) + Bengaluru uploads (100)
- **Labeling:** Gemini Pro 1.5 Flash

## Depth Statistics
```
mean:   42.3 cm
std:    28.4 cm
median: 38.5 cm
min:    0 cm (dry)
max:    150 cm (head-deep)
```

## Label Distribution
- 0 cm (dry): 18 images
- 1-20 cm (ankle): 32 images
- 20-50 cm (knee): 48 images
- 50-100 cm (waist): 38 images
- 100+ cm (head): 14 images

## Image Sources
| Source | Count | Type |
|--------|-------|------|
| Kaggle | 50 | Various global floods |
| Wikimedia | 30 | Public domain floods |
| User uploads | 100 | Bengaluru streets |

## Quality Assurance
✅ No label collapse (mean ≠ 0)
✅ All images RGB 224×224+
✅ No duplicates
✅ Diverse lighting conditions
✅ Multiple reference objects

## Usage
```python
from torchvision import transforms
from src.dataset import FloodDataset

dataset = FloodDataset(
    folder='datasets/training_v3/train',
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
)
```

## Model Performance
- **Model:** best_flood_model_water_aware.pth (v3)
- **Epochs:** 30
- **Best Val Loss:** 0.0342
- **MAE:** 5.2 cm
- **RMSE:** 8.7 cm

## For Retraining
See `Retrain_with_AutoDownload.ipynb` for workflow.
"""


# ═════════════════════════════════════════════════════════════════════════════
# SUGGESTED FOLDER STRUCTURE (Git)
# ═════════════════════════════════════════════════════════════════════════════

RECOMMENDED_STRUCTURE = """
flood-depth-estimator/
│
├── models/
│   ├── best_flood_model_water_aware.pth          [Main model, Git LFS]
│   ├── model_registry.json                       [Version tracker]
│   └── archive/
│       ├── v1_initial_collapse.pth               [Historical]
│       ├── v2_50_images.pth
│       └── v3_150_images.pth
│
├── datasets/
│   └── training_v3/                              [Versioned dataset]
│       ├── train/                                [127 images]
│       ├── val/                                  [23 images]
│       ├── labels.csv                            [CRITICAL]
│       ├── metadata.json                         [Stats + QA]
│       └── README.md                             [Documentation]
│
├── training_logs/                                [Experiment tracking]
│   ├── v3_training_history.json
│   ├── v3_loss_curve.png
│   └── v3_metrics.txt
│
└── Retrain_with_AutoDownload.ipynb
"""


# ═════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE: What to do with each file
# ═════════════════════════════════════════════════════════════════════════════

ASSET_MANAGEMENT = {
    "best_flood_model_water_aware.pth": {
        "size": "50 MB",
        "storage": "Git LFS + GitHub releases",
        "backup": "D: drive + Google Drive",
        "version": "model_registry.json",
    },
    "training_v3/ (images + labels)": {
        "size": "200-500 MB",
        "storage": "Option A: DVC, Option B: Google Drive, Option C: Archive as .zip",
        "backup": "Google Drive folder",
        "needed_for": "Retraining, model audits",
    },
    "model_registry.json": {
        "size": "< 1 KB",
        "storage": "Git repo root",
        "backup": "Auto (git)",
        "tracks": "All model versions, metrics, datasets",
    },
    "labels.csv": {
        "size": "< 100 KB",
        "storage": "datasets/training_v3/",
        "backup": "Git",
        "critical": "YES - defines training targets",
    },
}


if __name__ == "__main__":
    print("📊 MODEL & DATASET MANAGEMENT GUIDE")
    print("=" * 60)
    print("\n✅ Option 1: GitHub (simple, <200 images)")
    print("✅ Option 2: DVC (scalable, >500 images)")
    print("✅ Option 3: HuggingFace (public sharing)")
    print("\n🎯 RECOMMENDED: GitHub + DVC + metadata tracking")
