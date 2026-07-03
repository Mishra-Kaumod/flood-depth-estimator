# 📊 After Training: Asset Management Checklist

## You have 3 things to manage:
1. **Model** (50MB `.pth` file)
2. **Training Images** (100-300 JPGs)
3. **Labels** (CSV with depth values)

---

## 🚀 QUICK WORKFLOW (5 minutes)

### Step 1: Download from Colab
```
✓ best_flood_model_water_aware.pth  (50 MB)
✓ training_v3/                      (200-500 MB)
    ├── train/
    ├── val/
    └── labels.csv
```

### Step 2: Organize Locally
```bash
cd flood-depth-estimator

python organize_trained_assets.py \
  --model "Downloads/best_flood_model_water_aware.pth" \
  --dataset "Downloads/training_v3" \
  --version "v3"
```

This creates:
```
models/
├── best_flood_model_water_aware.pth    (current)
└── archive/
    └── v3_20260703_130000.pth          (backup)

datasets/
└── training_v3/
    ├── train/       (127 images)
    ├── val/         (23 images)
    ├── labels.csv
    ├── metadata.json
    └── README.md

model_registry.json                      (updated!)
```

### Step 3: Commit to Git
```bash
git add models/ datasets/ model_registry.json
git commit -m "Add trained model v3: 150 images, val_loss=0.0342"
git push origin main
```

### Step 4: Done! ✅
- Model deployed at `http://localhost:5000`
- Dataset versioned and documented
- Training history tracked in `model_registry.json`
- Server auto-reloads with new model

---

## 📋 Model Registry Format

**File:** `model_registry.json`

```json
{
  "models": [
    {
      "version": "v3",
      "date_trained": "2026-07-03T13:00:00+05:30",
      "model_path": "models/best_flood_model_water_aware.pth",
      "dataset_path": "datasets/training_v3",
      "checkpoint_info": {
        "epoch": 30,
        "best_val_loss": 0.0342,
        "is_collapsed": false
      },
      "dataset_stats": {
        "total_images": 150,
        "train_images": 127,
        "val_images": 23,
        "depth_mean": 42.3,
        "depth_std": 28.4
      }
    }
  ]
}
```

---

## 🗂️ Folder Structure Reference

### Models
```
models/
├── best_flood_model_water_aware.pth     ← Current (deployed)
├── model_registry.json                   ← Version tracker
└── archive/
    ├── v1_20260701_103000.pth           ← Failed (label collapse)
    ├── v2_20260702_140000.pth           ← First attempt
    └── v3_20260703_130000.pth           ← Current backup
```

### Datasets
```
datasets/
├── training_v1/                 ← First attempt (50 images, collapsed)
├── training_v2/                 ← Second attempt (100 images, not great)
└── training_v3/                 ← CURRENT ⭐ (150 images, good)
    ├── train/                   (127 images)
    │   ├── flood_001.jpg
    │   └── ...
    ├── val/                     (23 images)
    │   ├── flood_128.jpg
    │   └── ...
    ├── labels.csv               ← CRITICAL: depth labels
    ├── metadata.json            ← Auto-generated stats
    └── README.md                ← Auto-generated docs
```

---

## 🔄 Versioning Strategy

**Model versions:**
- `v1`: Initial (usually has label collapse)
- `v2`: First retrain (test)
- `v3`: Production (best metrics)

**Dataset versions:**
- Same naming: `training_v1`, `training_v2`, `training_v3`
- Keeps history → can retrain old versions if needed

**Always update model_registry.json:**
- What changed (more images, better labels, different model)
- Metrics (val_loss, MAE, RMSE)
- Dataset used (images count, sources)

---

## 📊 Deployment Checklist

```
☑️  Model checkpoint saved (50 MB)
☑️  Dataset organized in datasets/training_vX/
☑️  labels.csv exists (NOT empty, NOT all zeros)
☑️  metadata.json created with stats
☑️  model_registry.json updated
☑️  Git commit with both model + dataset
☑️  Git push origin main
☑️  Server restarted
☑️  Tested at http://localhost:5000
```

---

## ❌ Common Mistakes (Avoid!)

| Mistake | Fix |
|---------|-----|
| Losing images after training | Store in `datasets/training_vX/` |
| No labels.csv version control | Commit to Git (small file) |
| Model files in .gitignore | Use Git LFS for models/ |
| No version tracking | Use model_registry.json |
| Overwriting old models | Keep in models/archive/ |
| Unclear which dataset → which model | Use versioning (v1, v2, v3) |

---

## 📈 When to Create New Version

Create `training_v4` when:
- ✅ You collect 50+ new images
- ✅ You get more accurate labels (manual review)
- ✅ You use better labeling method (e.g., Gemini → human labeler)
- ✅ You want to A/B test different image sources

Don't create new version for:
- ❌ Just tweaking hyperparameters (same data = same training_v3)
- ❌ Different model architecture (same data, different model.pth)

---

## 🚀 For Production

When ready to ship:
1. Create GitHub Release with model
2. Upload to HuggingFace Hub (optional)
3. Keep datasets/ for reproducibility
4. Document everything in README.md

---

## 📞 Quick Help

**Where did my Colab files go?**
- Look in Google Drive → Colab downloads
- Or check browser Downloads folder

**How do I organize them?**
- Use `organize_trained_assets.py` script

**How do I track versions?**
- `model_registry.json` tracks everything

**What if training fails again?**
- Check datasets/training_v3/labels.csv → mean should NOT be 0
- Check metadata.json → zero_cm_count should be < 80% of total

---

## 📚 Full Guides

- **MODEL_DATASET_MANAGEMENT.md** — Complete best practices
- **organize_trained_assets.py** — Automation script
- **model_registry.json** — Version tracker
