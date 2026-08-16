# Flood Depth Model Training Guide

This guide explains how to retrain and test the flood-depth models in this project.

## Current Model Setup

The project uses multiple signals:

- **EfficientNet depth model**: supervised model trained from image + depth label.
- **Water detection / SegFormer-style stage**: estimates water coverage.
- **Depth Anything V2**: gives scene depth clues.
- **YOLO/reference objects**: detects cars, trucks, people, etc.
- **Manual pipeline rules**: safety gates, shallow-water gates, severity mapping.
- **Residual fusion model**: combines EfficientNet + pipeline evidence into the final depth.

The main trainable models are:

```text
EfficientNet depth model
Residual fusion model
```

## Important Rule

Do not retrain for every single wrong image.

Collect a batch of useful failure examples first:

```text
false water positive
dry road / dry field
wet road but no flood
shallow puddle
far-water-only road
deep flood underpredicted
vehicle-submersion overpredicted
```

Then label them and retrain as a batch.

## Folder Structure

Use these folders:

```text
training_data/images
training_data/un_labled_images
training_data/labels.csv
training_data/labels_aligned.csv
training_data/labels_clean.csv
training_runs/
models/candidate/
reports/
```

Meaning:

```text
training_data/images              labeled images only
training_data/un_labled_images    images not yet trusted for training
training_data/labels.csv          source labels
training_data/labels_clean.csv    clean labels used for training
training_runs/                    generated train/val/test splits
models/candidate/                 newly trained model files
reports/                          evaluation reports
```

## Step 1: Add Or Fix Labels

Only put images in `training_data/images` when they have a trusted depth label.

Example labels:

```text
dry image -> 0 cm
small puddle -> 2 to 5 cm
far water only -> 0 to 5 cm if near-camera road is dry
shallow road water -> 10 to 25 cm
vehicle wheel-level water -> 30 to 60 cm
deep vehicle submersion -> 60+ cm
```

After adding or fixing labels, make sure `labels_clean.csv` contains only files that actually exist in `training_data/images`.

## Step 2: Rebuild Train / Validation / Test Split

From the project root:

```powershell
cd C:\Users\abhij\OneDrive\Desktop\WellLabs\flood_project_cleaned
```

Create a fresh split:

```powershell
python scripts\prepare_training_and_splits.py --images-dir training_data\images --labels training_data\labels_clean.csv --n 10000 --train 340 --val 42 --test 43 --out training_runs\manual_retrain
```

Adjust `--train`, `--val`, and `--test` based on the number of labeled images.

Example:

```text
425 labeled images -> train 340, val 42, test 43
```

## Step 3: Train EfficientNet

EfficientNet is supervised. It learns from:

```text
image -> correct depth_cm
```

Run:

```powershell
python scripts\train_candidate_depth_model.py --train-images training_runs\manual_retrain\images\train --train-labels training_runs\manual_retrain\labels_train.csv --val-images training_runs\manual_retrain\images\val --val-labels training_runs\manual_retrain\labels_val.csv --base-model models\candidate\best_flood_model_water_aware_hardneg.pth --output models\candidate\best_flood_model_water_aware_new.pth --max-depth-cm 180 --epochs 14 --batch-size 16 --learning-rate 0.00008
```

This creates:

```text
models/candidate/best_flood_model_water_aware_new.pth
```

## Step 4: Evaluate EfficientNet

Compare the new model against the current active model:

```powershell
python scripts\evaluate_candidate_depth_model.py --images-dir training_runs\manual_retrain\images\test --labels training_runs\manual_retrain\labels_test.csv --production-model models\candidate\best_flood_model_water_aware_hardneg.pth --candidate-model models\candidate\best_flood_model_water_aware_new.pth --production-max-depth-cm 180 --candidate-max-depth-cm 180 --out reports\candidate_depth_eval_new.csv
```

Good result looks like:

```text
Production MAE: 9.02 cm
Candidate MAE: 6.40 cm
Candidate better on 27 of 42 images
```

Bad result looks like:

```text
Production MAE: 6.40 cm
Candidate MAE: 7.41 cm
Candidate better on 16 of 42 images
```

Only promote EfficientNet if:

```text
candidate MAE is lower
candidate is better on more than half of test images
deep flood examples did not get worse badly
dry/no-water examples improved
important known failure examples improved
```

## Step 5: Temporarily Point Config To New EfficientNet

Only do this if you want to train fusion using the new EfficientNet signal.

Edit `config/config.yaml`:

```yaml
efficientnet_signal:
  model_path: "models/candidate/best_flood_model_water_aware_new.pth"
```

If the new EfficientNet is worse overall, do not promote it. Keep:

```yaml
efficientnet_signal:
  model_path: "models/candidate/best_flood_model_water_aware_hardneg.pth"
```

## Step 6: Train Residual Fusion

Residual fusion learns how to combine:

```text
EfficientNet depth
water coverage
near/mid/far water
Depth Anything
YOLO/reference object signals
manual pipeline depth
safety flags
```

Run:

```powershell
python scripts\train_residual_fusion_depth_model.py --rebuild-features --split-dir training_runs\manual_retrain --features-out reports\fusion_training_features_new.csv --model-out models\candidate\residual_fusion_depth_model_new.pt --report-out reports\residual_fusion_depth_eval_new.csv --progress-every 50 --epochs 250 --patience 35 --print-every 10
```

This creates:

```text
models/candidate/residual_fusion_depth_model_new.pt
reports/residual_fusion_depth_eval_new.csv
```

## Step 7: Evaluate Fusion

The fusion training output prints lines like:

```text
test MAE | residual_fusion=6.28cm | current_pipeline=28.07cm | efficientnet=6.40cm
```

Good result:

```text
residual_fusion MAE is lower than current active fusion
residual_fusion MAE is lower than raw pipeline
residual_fusion does not damage important deep/shallow examples
```

Bad result:

```text
new fusion MAE is worse than current active fusion
worst errors become dangerous
dry/no-water cases still fail badly
deep flood images are underpredicted
```

## Step 8: Promote Models

Only update `config/config.yaml` after evaluation proves the model is better.

Promote EfficientNet:

```yaml
efficientnet_signal:
  model_path: "models/candidate/best_flood_model_water_aware_new.pth"
```

Promote fusion:

```yaml
residual_fusion_signal:
  model_path: "models/candidate/residual_fusion_depth_model_new.pt"
```

It is okay to promote only fusion and keep the old EfficientNet if that is better overall.

That happened in our latest training:

```text
New EfficientNet was worse overall, so it was not promoted.
New fusion model was slightly better, so only fusion was promoted.
```

## Step 9: Test Important Images

Run known examples:

```powershell
python main.py image training_data\images\image_75.jpg --storage local
python main.py image training_data\images\image_80.jpg --storage local
python main.py image training_data\images\image_407.jpg --storage local
python main.py image training_data\images\image_408.jpg --storage local
python main.py image training_data\images\image_432.jpg --storage local
python main.py image training_data\images\image_433.jpg --storage local
python main.py image training_data\images\image_439.jpg --storage local
```

Also test external images:

```powershell
python main.py image external_test_images\your_image.jpg --storage local
```

## How To Decide If Model Is Better

Use this checklist:

```text
1. Candidate MAE is lower than current model.
2. Candidate is better on more than half of test images.
3. Worst errors are not dangerous.
4. Deep flood images are not underpredicted badly.
5. Dry/no-water images move closer to 0 cm.
6. Shallow puddles do not become 50+ cm.
7. Far-water-only road images do not trigger emergency severity.
8. Known problem images improved.
```

If the model fixes one image but hurts many others, do not promote it.

## Current Active Status

At the time this guide was written, active config uses:

```text
EfficientNet:
models/candidate/best_flood_model_water_aware_hardneg.pth

Residual fusion:
models/candidate/residual_fusion_depth_model_retrain_old_eff_20260816_231536.pt
```

Current status:

```text
EfficientNet hard-negative model is still best overall.
Latest EfficientNet retrain fixed some dry/no-water examples but hurt overall accuracy.
Latest fusion retrain improved test MAE slightly and was promoted.
Dry/no-water false positives are known limitation and need more negative examples.
```

## Notes About LLM Judge

LLM judge is optional.

It is useful as a final reviewer, but it requires API key setup:

```text
GOOGLE_API_KEY
```

If the key is missing, output will show:

```text
LLM judge unavailable
```

The CV/model pipeline still runs without LLM judge.

