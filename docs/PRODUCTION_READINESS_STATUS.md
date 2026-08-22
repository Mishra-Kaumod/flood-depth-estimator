# Production Readiness Status

## Current Production Path

The service uses the staged fusion pipeline:

1. Water-region detection
2. YOLO/reference-object detection when available
3. Depth Anything V2 or dense-depth proxy
4. Fusion/calibration logic
5. EfficientNet-B0 candidate depth signal for severe-underestimate correction
6. Severity/action decision

The active production configuration is in `config/config.yaml`.

## Active Model Artifacts

- Primary configured depth checkpoint: `models/best_flood_model_water_aware.pth`
- Production candidate signal: `models/candidate/best_flood_model_water_aware.pth`
- Legacy severity checkpoint: `severity_model.pth`
- YOLO reference-object weights: `yolov8n.pt`

## Candidate Validation Summary

Held-out test split:

- Train: 638 images
- Validation: 80 images
- Test: 80 images

Evaluation report:

- `reports/candidate_depth_eval.csv`

Results on the held-out 80-image test set:

- Existing checkpoint MAE: 105.174 cm
- Candidate checkpoint MAE: 24.800 cm
- Candidate better on: 80 of 80 images

Known limitation:

- The held-out test split is skewed toward high/critical depth images.
- The candidate still underpredicts some very deep cases above 110 cm.
- Human review is required before using these predictions for safety-critical automated decisions.

## Production Corrections

The EfficientNet candidate can correct the fusion result only when all of these are true:

- Candidate depth is at least 60 cm
- Fusion depth is below 50 cm
- Difference is at least 25 cm
- Water coverage is broad
- Near-field water is present
- Immediate risk is detected

When applied, the output retains:

- `pre_correction_depth_cm`
- `efficientnet_candidate_depth_cm`
- `efficientnet_correction_applied`
- `correction_reason`

## Required Before Client Delivery

- Run endpoint smoke tests on representative client images.
- Review `production_samples/review_worst` manually.
- Decide whether to promote `models/candidate/best_flood_model_water_aware.pth` to the primary checkpoint.
- Confirm deployment environment has required dependencies installed.
- Configure `GOOGLE_API_KEY` only if LLM judge corrections are intended for production.
