# Flood Depth Model Storage

This folder contains the active trained depth checkpoint copied into the cleaned
project.

## Active Model

- **File:** `models/best_flood_model_water_aware.pth`
- **Architecture:** EfficientNet-B0 + water-aware regression head
- **Input:** 224x224 RGB image
- **Output:** Normalized flood depth regression
- **Framework:** PyTorch
- **Use:** Default retraining checkpoint and fallback single-frame inference model

The cleaned project's default inference mode is the staged fusion pipeline:

`water detection -> YOLO/reference objects -> dense depth proxy/Depth Anything -> fusion -> severity`

That pipeline is configured in `config/config.yaml` as:

```yaml
inference:
  model_path: "models/best_flood_model_water_aware.pth"
  pipeline_mode: "segformer_yolov8_depthv2_fusion"
```

## Safe Checkpoint Inspection

Use `weights_only=True` when inspecting untrusted PyTorch checkpoints:

```python
import torch

checkpoint = torch.load(
    "models/best_flood_model_water_aware.pth",
    map_location="cpu",
    weights_only=True,
)
state_dict = checkpoint.get("model_state_dict", checkpoint)
print(len(state_dict))
```

## Retraining

The copied retraining utilities are in `scripts/`:

- `prepare_training_and_splits.py`
- `retrain_flood_classifier.py`
- `validate_training_labels.py`
- `shadow_canary_check.py`

The labeled training data is under `training_data/`.
