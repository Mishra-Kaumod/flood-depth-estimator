# Flood Depth Model Storage

This directory is reserved for trained model checkpoints.

## Model File Storage Strategy

### For Development (Local)
Models are NOT committed to GitHub to keep repository size small.

When training locally:
```bash
python src/train.py --config config/config.yaml --output models
```
- Saves best model to: `models/best_flood_model.pth` (not in git)

### For Production (AWS S3)
Models are stored in S3 for:
- Version control
- Easy sharing across team
- Automatic download on deployment

To upload trained model to S3:
```bash
aws s3 cp best_flood_model.pth s3://your-flood-bucket/models/
```

### For GitHub Codespace
Model downloads automatically on first run:
```python
from src.train import build_model, load_model_from_s3
model = load_model_from_s3("best_flood_model.pth")  # Downloads if not found
```

## Model Specifications
- Architecture: EfficientNet-B0
- Input: 224x224 RGB image
- Output: Depth in cm (0-100)
- Size: ~20 MB
- Framework: PyTorch

## Best Practices
1. ✅ Never commit large model files to GitHub
2. ✅ Use Git LFS only if model must be version-controlled
3. ✅ Store production models in S3/cloud storage
4. ✅ Version models with timestamps or git tags
5. ✅ Keep .pth files in .gitignore

## AWS S3 Setup
```bash
# Create S3 bucket for models
aws s3 mb s3://flood-depth-models

# Upload model
aws s3 cp models/best_flood_model.pth s3://flood-depth-models/v1/

# List models
aws s3 ls s3://flood-depth-models/
```
