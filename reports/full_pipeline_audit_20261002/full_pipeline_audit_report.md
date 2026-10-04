# Full Production Flood-Depth Pipeline Audit

**Scope:** frozen production configuration and checkpoints, static source review, checkpoint metadata inspection, and live development traces from five approved clean-validation rows plus three approved clean-training rows. No model was retrained. The consumed internal-test and external-challenge sets were not read.

## Execution Contract Observed

The live path is `src/pipeline.py -> SegformerYoloDepthV2Pipeline.predict()` in `src/segformer_yolo_depthv2_pipeline.py`.

Actual order:

1. RGB image validation.
2. `WaterRegionDetector.detect()` under a method/stage named `SegFormer`.
3. no-water guard, wet-road/no-water guard, road-scene classifier, and depth-regime classifier inference.
4. EfficientNet candidate depth and optional muddy-water fallback.
5. object/reference detection, then Depth Anything V2 dense depth.
6. separate `ReferenceDepthEstimator.estimate()` and physical feature fusion.
7. mask-conditioned candidate, calibration/base depth, EfficientNet correction, model-agreement engine.
8. pre-residual no-water/scene cap calculation, then V14 residual fusion.
9. dynamic broad-mask resolver, mask-conditioned high-flood correction, disabled strong-deep correction, dry-land guard, final no-water guard, and final dry-road zero override.

The final owner is therefore **not V14 alone**. Any later enabled branch can replace its output.

## Checkpoint and Preprocessing Verification

| Stage | Loaded artifact / behavior | Audit result | Finding class |
| --- | --- | --- | --- |
| RGB entry | PIL RGB in `src/pipeline.py` | Correct handoff to NumPy RGB | Pass |
| Water mask | `WaterRegionDetector`, not a SegFormer checkpoint | Stage name is misleading; no SegFormer model is loaded | 4 pipeline-order/branch problem |
| No-water guard | `no_water_guard_teammate_augmented.pth`, SHA `0dbf...587d`; MobileNetV3, 224, ImageNet normalize | Checkpoint class order is `no_water, water`; inference uses the same normalization as its trainer | Pass, but model limitation remains |
| Wet-road guard | `wet_road_no_water_guard_test0916.pth`, SHA `85e4...efe3`; same two-class contract | Loaded correctly | Pass, but model limitation remains |
| Road scene | `road_scene_classifier_4class_shallow_v2.pth`, SHA `3763...55b9`; class order verified | 224/ImageNet validation transform matches trainer | Pass, but classifier has known wet/shallow errors |
| EfficientNet | `best_flood_model_water_aware_hardneg.pth`, SHA `78bd...fb40`; checkpoint max depth 180 | Model architecture and `224 + ImageNet normalize + sigmoid * 180` match checkpoint metadata | Pass |
| Dense depth | Depth Anything V2 configured as `depth-anything/Depth-Anything-V2-Small-hf` | It is monocular relative depth, normalized independently per image, then treated as `p90 * 120 cm` | 3 feature/unit calculation error |
| Mask-conditioned model | `FloodDepth-MaskConditionedFusion.pth`, SHA `e681...ba862`; 384 ImageNet normalize, log1p inverse | Load/target transform are handled correctly. Checkpoint validation R2 is only 0.297. | 6 unreliable individual model |
| V14 residual | `residual_fusion_depth_model_ankle_v14_no_leak.pt`, SHA `f4a8...bb8` | Checkpoint supplies the active 40 feature names, means/stds, base feature `pre_residual_output_cap_capped_depth_cm`, and max residual 120 | Pass for loading/contract |
| Dynamic resolver | `depth_regime_classifier_v6_targeted_conservative.pt`, SHA `f129...b8fd` | Loads the expected four classes and transform | Pass for loading; later override remains risky |

The configured Depth Anything loader issued an unauthenticated Hugging Face request during audit. A deployment without a cached model or network can silently use `dense-depth-proxy`; that is an environment-dependent pipeline branch and must be recorded in deployment health/trace metadata.

## Confirmed Pipeline Findings

### F-01: “SegFormer” is not SegFormer

`_segformer_water_mask()` calls `WaterRegionDetector.detect()` and the live trace records `classical-water-detector`. No SegFormer weights are loaded. This is a **4 pipeline naming/branch problem**, not a checkpoint-loading failure. It matters because users and evaluation reports may attribute failures to a segmentation model that is not running.

### F-02: Reference count and reference depth do not describe the same evidence

The live pipeline obtains `reference_count` and submersion values from YOLO/object-detector boxes, but calls `ReferenceDepthEstimator.estimate(image_rgb)` separately to create `reference_depth_cm`. That estimator builds its own HSV/contour mask and its own contour references. Thus `reference_count=8` does not mean eight references supported `reference_depth_cm=125.9`.

This is a **3 feature/calculation error** and a direct explanation for contradictory inputs to fusion and V14. It was visible in the live 4.5 cm trace: eight YOLO references, a 125.9 cm reference estimate, 27.67 cm EfficientNet estimate, and a 4.5 cm label.

### F-03: Multiple-reference aggregation is not actually implemented for depth

`ReferenceDepthEstimator` sorts contour candidates but `_estimate_from_vehicle()` and `_estimate_from_person()` return the first usable object. A vehicle and person are averaged only if both first-object estimates exist. YOLO boxes are not converted into independent depth estimates or robustly aggregated.

This is a **3 feature/calculation error**. A single large contour/reference can dominate the reported `reference_depth_cm`; confidence/disagreement is not propagated as an explicit reliability score.

### F-04: Dense-depth pseudo-centimetres are mixed with centimetre estimates

Depth Anything output is min-max normalized per image. Fusion computes `dense_depth_cm = percentile(depth_map, 90) * 120`; residual features use the same engineered scale. This value is useful as a relative cue but is not metric centimetres and is not comparable across images without calibration.

This is a **3 feature/unit error**. It should be named as a normalized proxy until a learned/calibrated mapping is proven.

### F-05: Semantic probabilities have four separate routes to alter numerical depth

The same semantic outputs are used in V14 features, pre-residual wet/shallow caps, final no-water zeroing, and a final dry-road override. They also affect dynamic-resolver eligibility. This is **5 conflicting rule design**: a classifier error can affect centimetres repeatedly rather than once as an uncertainty signal.

The road-scene trace calls itself `advisory`, but config has `advisory_only: false`, wet-road capping enabled, and direct dry-road zeroing enabled. The trace label is therefore inaccurate.

### F-06: V14’s effective correction bound is bypassable

V14 has a nominal `max_live_adjustment_cm=120`, but a large adjustment is passed through whenever the residual output is close to its base feature (`allow_large_candidate_aligned_adjustment`). The correction is then measured against the pre-residual pipeline depth, which can differ from the residual base because pre-output caps are recorded rather than applied before V14.

This is **4 pipeline-order/branch problem**. It explains a live `109.53 -> 18.14 cm` correction (-91.39 cm) on the 50 cm representative.

### F-07: Later branches can erase or replace V14

After V14, the dynamic resolver can increase depth; the mask-conditioned high-flood correction can replace it; dry-land/no-water guards can force zero; and the dry-road classifier can force zero. The configured `mask_conditioned_fusion_signal.use_for_final_decision: false` does **not** disable `_apply_mask_conditioned_high_flood_correction`; that method instead checks `high_flood_correction_enabled: true`.

This is a confirmed **4 pipeline-order/branch problem** and a configuration-semantics defect. `use_for_final_decision` is misleading/dead for this path.

### F-08: Trace order and values are misleading

The residual trace entry is appended after all final guards and reports the then-current final `depth_cm`, not the depth immediately after V14. Likewise, the final `Calibration/Severity Model` trace entry is appended last, although calibration ran much earlier. Existing traces cannot reliably establish handoff order without structured feature reconstruction.

This is an **1 implementation/observability bug**. The added development trace script records explicit stage fields, but production trace semantics still need correction before operational reliance.

### F-09: Missing values are silently converted to zero in residual features

`_feature_float()` maps missing, nonnumeric, and non-finite values to `0.0`; boolean conversion uses Python truthiness. Live tensors are finite, but absent signals become indistinguishable from genuine zero evidence. A serialized string such as `"False"` would evaluate as true if it ever entered a boolean feature path.

This is an **1 implementation bug** / **3 feature-contract risk**. It was previously avoided in the current live path because values are native Python numeric/bool values, not because the contract rejects ambiguity.

## Representative Live Traces

Artifacts: `reports/full_pipeline_audit_20261002/*_trace.json`.

| Image | Actual | Physical/base path | V14 / final result | First major failure |
| --- | ---: | --- | --- | --- |
| `image_4.jpg` | 35.0 | mask 0.57%, EfficientNet/base 22.95, dense proxy 34.36, unrelated reference estimate 108.9 | V14 -22.95 -> final 0 | Water mask and semantics describe dry/no water; V14 completes collapse |
| `image_263.jpg` | 25.0 | mask 30.25%, near 28.57%, EfficientNet/base 51.37, dense proxy 90.35, reference 108.4 | V14 -51.37 -> final 0 | Physical evidence is meaningful but road scene says dry and wet-road guard gives 0.802; V14 collapse |
| `abhinav_20260923_07_50CM.jpeg` | 50.0 | mask 60.32%, near 53.07%, EfficientNet 52.09, base 109.53, reference 119.1 | V14 -91.39 -> final 18.14 | Base is reference-driven high; V14 over-corrects far below the reliable EfficientNet cue |
| `abhinav_20260923_16_4.5cm.jpeg` | 4.5 | mask 58.41%, eight YOLO refs, reference 125.9, EfficientNet 27.67, base 65 | V14 -33.07 -> final 31.93 | Reference/mask physical base already severely high; residual only partially recovers |
| `abhinav_20260923_01_55cm.jpg` | 55.0 | mask 64.52%, near 99.75%, EfficientNet/base 45.11/45.0, reference 125.9 | raw V14 collapse intercepted by existing production moderate-flood fallback; final 45.07 | Existing fallback works here, but is narrow and not generalizable (rejected separately on internal evaluation) |
| `47.0cm.png` (train representative) | 47.0 | mask 3.84%, near 0%, EfficientNet 49.35, reference 112.6 | low-water gate, V14 skipped, final 5.0 | Segmentation/low-water gate is decisive before residual |
| `25.04cm.png` (train representative) | 25.04 | mask 5.49%, near 1.61%, base 45.0, EfficientNet 28.39 | V14 -> 30.17 | V14 gives a useful moderate correction despite reference conflict |
| `72.5cm.png` (train representative) | 72.5 | mask 50.48%, near 91.92%, EfficientNet 76.51 | residual -> 75.11 | Correct deep behavior; physical cues agree |

## Classifier Audit

The input model contracts match their production trainer transforms: 224x224 RGB, ImageNet normalization, MobileNetV3 Small, and checkpoint class ordering. The issue is not a class-index reversal.

Known limitation: the road-scene checkpoint’s archived test metrics show wet-road recall 54.5% and shallow-flood recall 54.5%. The current clean classifier development work also found wet road can be classified as dry or shallow. This is **6 unreliable individual model / expected distribution limitation**, made operationally severe by F-05’s repeated numeric use.

## Fusion/Resolver Audit

- The base calibration is rule-heavy, contains overlapping hard caps, and uses a 0.65 reference / 0.35 dense blend whenever any reference exists. It has no reference reliability score.
- The model-agreement engine mixes a calibrated base, EfficientNet metric candidate, normalized dense proxy, reference estimate, and experimental mask-conditioned signal using hand-selected weights and a 25 cm tolerance.
- The dynamic resolver uses a separate depth-regime classifier and rescales signals before percentile fusion. It can only increase output, so it is asymmetric and cannot repair reference-induced overestimation.
- The mask-conditioned high-flood correction can use a model whose stored validation R2 is 0.297 to replace a later result when its conditions pass.

These are **5 conflicting rule design** plus **6 individual-model reliability** concerns, not evidence of a single checkpoint load failure.

## Offline Versus Live Parity

Residual-only offline evaluation is not equivalent to live inference because live prediction includes model agreement, pre/post guards, dynamic resolver, high-flood correction, dry-land/no-water zeroing, and dry-road override. The V14 model itself uses the checkpoint’s 40-name contract correctly, but a residual-only table cannot reproduce the final live depth.

The live trace of the 55 cm representative demonstrates this: a raw V14 collapse was prevented by the currently enabled production moderate-flood safety fallback, yielding 45.07 cm. Earlier residual-only reconstruction cannot represent that stage unless it explicitly replays the full pipeline.

## Recommended Minimal Fixes (Not Implemented)

1. **Truthful stage contracts:** rename the current segmentation stage/backend; record explicit post-stage depths in order; make `use_for_final_decision` either functional or remove it. Fix trace ordering before any model work.
2. **Reference consistency:** calculate reference depth from the same detected objects counted in `reference_count`, retain per-object estimates, and introduce a reliability/dispersion score before a reference participates in base fusion.
3. **Units:** label monocular depth as normalized proxy and calibrate it separately before mixing it numerically with centimetres.
4. **Single ownership:** consolidate semantic rules so they can set regime/confidence and veto unsafe refinement but do not independently cap/zero depth at multiple stages.
5. **Explicit handoff contract:** make the pre-residual base exactly the numerical input to V14; record raw V14 output, bounded applied output, and every subsequent replacement independently.
6. **Fail closed for features:** distinguish unavailable from zero and reject non-finite/string booleans before residual inference.

These are audit-backed foundations for the proposed modular architecture. They should be developed on new group-locked data only; the consumed internal and external sets remain reporting-only.
