# DeepLab Training Plan — Municipality Flood Intelligence Platform

## Goal
Train a DeepLabV3-based water segmentation model using FloodNet only as the Phase 1 dataset.

This document defines the dataset structure, label mapping, model architecture, hyperparameters, augmentation, metrics, training schedule, and expected performance for the first production-ready segmentation baseline.

## 1) Dataset structure

Use a canonical FloodNet-style folder layout for training and validation.

Recommended structure:
- `dataset_labeled/`
  - `Image/` — original RGB images
  - `Mask/` — corresponding ground-truth water masks
  - `metadata.csv` — `Image,Mask` pairs

For training, split the available images into:
- `train/` — 70% of samples
- `val/` — 20% of samples
- `test/` — 10% of samples reserved for final evaluation

Suggested output structure after preprocessing:
- `data/floodnet/train/images/`  
- `data/floodnet/train/masks/`  
- `data/floodnet/val/images/`  
- `data/floodnet/val/masks/`  
- `data/floodnet/test/images/`  
- `data/floodnet/test/masks/`

Each image file should be paired with a same-stem mask file (e.g. `0001.jpg` and `0001.png`).

## 2) Label mapping

For Phase 1, treat FloodNet as a binary water segmentation task.

Mapping rules:
- All non-zero mask values → water class (label `1`)
- Zero values → background / non-water class (label `0`)

Rationale:
- Existing FloodNet mask values are multi-valued and may encode depth or palette indices.
- Binary mapping simplifies the first DeepLab baseline and avoids class explosion.

Preprocessing steps:
- Read each mask in grayscale.
- Convert to binary with `mask = (mask > 0).astype(np.uint8)`.
- Optionally multiply by `255` for uint8 PNG storage.
- If image/mask dimensions differ, resize the mask to the image resolution using nearest-neighbor interpolation.

## 3) DeepLabV3 architecture

Use `torchvision` DeepLabV3 with a ResNet backbone.

Recommended baseline architecture:
- `torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)`
- Set `num_classes=2` for binary water segmentation.
- Use pretrained ImageNet backbone weights.

Production-ready upgrade path:
- If resources allow, move to `deeplabv3_resnet101` for stronger feature representation.
- Export later to TorchScript or ONNX for inference.

## 4) Hyperparameters

Baseline configuration:
- Batch size: 8–16 (depending on GPU memory)
- Learning rate: 1e-4
- Weight decay: 1e-4
- Optimizer: AdamW or SGD with momentum 0.9
- Scheduler: cosine annealing or ReduceLROnPlateau
- Loss: `CrossEntropyLoss` with class weights if water is under-represented
- Input size: 512×512 or 640×640
- Number of epochs: 30–50
- Validation frequency: every epoch

Notes:
- If the dataset is small, use a lower learning rate (1e-5) when fine-tuning.
- Use mixed precision if available to reduce GPU memory and speed training.

## 5) Augmentation strategy

Use strong image augmentation to improve robustness on street-level flood imagery.

Core augmentations:
- Random horizontal flip
- Random scale jitter (0.8–1.2)
- Random crop to model input size
- Random brightness/contrast adjustment
- Random blur / motion blur
- Color jitter (hue, saturation)

Safety checks:
- Apply identical spatial transforms to image and mask.
- Use nearest-neighbor resampling for masks.

Optional augmentations:
- Random fog/haze overlay
- Random occlusion / object cutout
- Gaussian noise or JPEG compression noise

## 6) Evaluation metrics

Primary metrics for Phase 1:
- Mean Intersection over Union (mIoU) for the water class and background
- Binary IoU or Jaccard index for the water mask
- Pixel accuracy for water vs background

Secondary metrics:
- Precision and recall for water pixels
- F1-score for the water class
- Validation loss trend

Accept final model when:
- water IoU > 0.75 on validation
- water precision and recall are both > 0.70
- test IoU is within 2–3% of validation IoU

## 7) Training schedule

Phase 1 schedule:
1. Data preparation (1–2 days)
   - Canonicalize masks to binary
   - Fix size mismatches
   - Create train/val/test splits
2. Baseline training experiment (2–3 days)
   - Train DeepLabV3-ResNet50 with ImageNet backbone
   - Track validation metrics and check for overfitting
3. Model refinement (2–3 days)
   - Tune augmentations and scheduler
   - Evaluate ResNet101 if needed
4. Final validation and export (1–2 days)
   - Freeze best checkpoint
   - Export TorchScript/ONNX for inference
   - Run final test set evaluation

Total Phase 1 timeline: ~1–2 weeks for a pilot-ready baseline, depending on compute availability.

## 8) Expected accuracy

Given FloodNet-only data and a binary mapping, target a pilot baseline in these ranges:
- Validation water IoU: 0.70–0.80
- Test water IoU: 0.68–0.78
- Pixel accuracy: 0.85–0.92
- Precision/Recall: 0.70–0.80

If the dataset is very small or highly imbalanced, use these values as a baseline target rather than a production guarantee.

### Notes on expectations

- Use strong augmentation and transfer learning to maximize generalization.
- If masks are noisy or if the dataset contains domain shift, prioritize validation stability over absolute accuracy.
- For a municipal flood intelligence platform, a Phase 1 DeepLab baseline should be valued for consistent water-mask quality and low false positives rather than raw IoU alone.

## Appendix: practical guidelines

- Prefer a 70/20/10 split where the validation set contains images from different flood events than the training set.
- Reserve a small municipality-specific holdout set for final acceptance testing.
- Keep a log of the best checkpoint with associated dataset split, loss, IoU, and model version.
- Document the label conversion rules clearly so future depth or multi-class extensions can reuse the same preprocessing.
