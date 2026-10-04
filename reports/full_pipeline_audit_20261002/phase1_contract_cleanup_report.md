# Phase 1 Pipeline Contract Cleanup

## Scope

Completed only Steps 1-4 from the approved cleanup plan: truthful trace/health metadata, diagnostic reference-object evidence, relative-depth naming, and non-enforcing V14 input-contract validation. No model, threshold, fusion weight, guard, resolver, or reference-depth contribution was changed.

## Files Changed

- `src/segformer_yolo_depthv2_pipeline.py`
- `src/pipeline.py`
- `scripts/audit_full_pipeline_trace.py`

## Changes And Prediction Effect

| Change | Classification | Prediction-neutral |
| --- | --- | --- |
| Renamed trace labels from `SegFormer` to `Water Segmentation`, YOLO to `Reference Detection`, and Depth Anything to `Relative-Depth Proxy (Depth Anything V2)`. | Implementation correction | Yes |
| Added ordered immutable `pipeline_stage_outputs` snapshots after semantic scoring, physical fusion, calibration, EfficientNet correction, model agreement, V14, resolver, guards, and final override. | Implementation correction | Yes |
| Corrected trace wording so road-scene is `scored`, residual reports its immediate applied depth, and final severity mapping is no longer called the calibration stage. | Implementation correction | Yes |
| Passed `pipeline_stage_outputs` through the unified event wrapper and audit script. | Implementation correction | Yes |
| Added per-YOLO-object diagnostic fields: nominal height, waterline depth proxy, detector/area/submersion reliability components, and explicit `used_by_current_reference_fusion: false`. | Feature/unit diagnostic | Yes |
| Labeled dense depth as a `per_image_normalized_relative_proxy`; retained the legacy `p90 x 120` numerical use unchanged and marked it non-metric. | Feature/unit correction | Yes |
| Added V14 feature-contract diagnostics for missing/non-finite/type-invalid inputs. Optional absent flags are explicitly documented as false defaults. Diagnostics are not enforced in Phase 1. | Implementation correction | Yes |

## Parity Check

Compared pre-Phase-1 and Phase-1 live traces for eight approved clean examples:

- Validation: `image_4.jpg`, `image_263.jpg`, `abhinav_20260923_01_55cm.jpg`, `abhinav_20260923_07_50CM.jpeg`, `abhinav_20260923_16_4.5cm.jpeg`
- Train: `47.0cm.png`, `25.04cm.png`, `72.5cm.png`

All eight final predictions matched exactly: **8/8, maximum difference 0.00 cm**. No unexpected output difference occurred. The recorded comparison is `phase1_parity_summary.json`.

Example handoff for `image_4.jpg`:

`calibration/base 22.95 -> model agreement 22.95 -> V14 applied 0.00 (delta -22.95) -> resolver 0.00 -> guards 0.00 -> final 0.00`

The full ordered example is `phase1_final_trace_example.json`.

## Frozen Artifacts Confirmed

- `config/config.yaml`: unchanged SHA-256 `6a5de50beb813fea30eba23cb460f455199276954d8c91d84ffec0cfca2ba7dd`
- Production V14 checkpoint: unchanged SHA-256 `f4a8dd74aa1b27f04051b33b28ce76bb7dd08cc097b512b3c90b20317a7f0bb8`

No internal-test or external-challenge data was accessed. No training, threshold tuning, feature extraction batch, or model download was performed intentionally; the already configured Depth Anything runtime loaded its cached model during the approved parity replay.

## Deferred

The current reference depth still comes from the old separate contour estimator, and the legacy dense proxy still contributes numerically. Replacing either, consolidating semantic numerical ownership, or changing base/residual handoff belongs to later approved phases only.
