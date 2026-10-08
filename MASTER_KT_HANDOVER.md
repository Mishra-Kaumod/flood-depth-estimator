# Master Knowledge Transfer / Handover — Flood Depth Project

Architecture updated on: 2026-10-08. Branch: `flood-model-v6-shadow-dev`. Shared-entrypoint commit: `7508638`; cleanup commit: `fd60fde`. This is the existing single master project handover, updated in place in Markdown and Word. Historical sections for the Frozen Predecessor Pipeline (V4) and Legacy Production/Reference Pipeline (V5) remain reference material. The workspace contains unrelated pre-existing uncommitted changes and local artifacts; this dated architecture record is not a clean-release or production-certification claim.

**Current Final Architecture (V6) is the only active prediction architecture. Legacy Production/Reference Pipeline (V5) is retained only for historical/reference purposes and is not part of normal inference. All normal entrypoints use the Shared V6 Inference Layer (V6).**

**V6 is the current architecture direction, not final production accuracy.** Its single metric-depth owner is EfficientNet: `primary_depth_cm → final_shadow_depth_cm`, with numerical owner `efficientnet_primary_anchor`. Final V6 training on the new dataset has not yet been completed.

**V5 is the previous controlled-testing/readiness baseline, not a solved flood-depth estimator.** It freezes Residual Fusion Model (V14) prediction behavior and adds Phase 1 observability/contract diagnostics. The tabular shallow-versus-meaningful router and MobileNetV3 pixel/image router were rejected and are not in the active V5 prediction path.

Read this document for navigation, source/config for executable behavior, and linked reports for evidence. Older supporting guides contain obsolete examples; their historical test workflows do not authorize reuse of consumed evaluation sets. The V6-default migration includes bounded execution-parity verification. No accuracy evaluation, retraining, checkpoint replacement or numerical-model change was performed.

### Architecture terminology

| Descriptive name | Internal version | Meaning |
| --- | --- | --- |
| Legacy Production/Reference Pipeline | V5 | Previous multi-stage pipeline retained as baseline/reference |
| Residual Fusion Model | V14 | Residual correction model used inside the legacy pipeline |
| Current Final Architecture | V6 | Current single-owner architecture using EfficientNet as primary metric depth |
| Shared V6 Inference Layer | V6 | Common inference path used by UI, CLI and video |
| V6 Video Input Pipeline | V6 | Video decoding and saved-JPEG/reload-RGB orchestration feeding the shared inference layer |
| Frozen Predecessor Pipeline | V4 | Earlier frozen pipeline retained for historical parity comparisons |

“Current Final Architecture” identifies the current architecture direction; it does not imply completed new-data training, final production accuracy or deployment acceptance. “Legacy Production/Reference Pipeline” names the retained baseline and does not certify historical production readiness. V14 is a model inside the legacy pipeline, not a separate application version. V6 names both the architecture and its shared inference layer; the descriptive name distinguishes their scope. Shorthand versions may be used after their descriptive meaning is established within the same section. Exact filenames, identifiers, code excerpts and the application title retain their original spelling for traceability.

## 1. Project Overview

The project estimates flood-water depth from photographs or sampled video frames, especially road/camera scenes. Current Final Architecture (V6) uses supervised EfficientNet depth as the only metric owner. Water masks and semantics are context; object/contour signals are diagnostic; Depth Anything is relative/context only. V6 returns primary and final centimetres with reporting-only reliability, uncertainty and stage evidence. The residual model (V14), resolver and guards do not overwrite V6 final depth. The useful Frozen Predecessor Pipeline (V4)/Legacy Production/Reference Pipeline (V5) fusion, severity and application history is retained below and explicitly scoped to that previous baseline.

The shared input contract of the Shared V6 Inference Layer (V6) is one PIL-decoded uint8 RGB image. `V6ShadowPipeline.predict()` returns `V6ShadowResult`, including `primary_depth_cm` and `final_shadow_depth_cm`; the image UI displays `final_shadow_depth_cm`; the shared response also exposes `primary_depth_cm` and `numerical_owner`. The previous V5 core returns `depth_cm`, confidence and stage evidence; its event wrapper expresses metres and its API may aggregate, review or mark depth unavailable. Those V5 policies are separate from V6.

**35 cm means an estimated scene flood-water depth of 0.35 metres under the project label convention.** It is not camera distance, pixel height, water coverage, or a calibrated per-pixel map. The system does not establish a surveyed measurement point or uniform depth across the scene. Partial-road/far-water images need explicit measurement-location labels. The old README's fixed severity midpoint example is not the current estimator. Confidence is a pipeline score, not a validated probability that the numerical depth is correct.

## 2. Repository Structure

```text
flood_project_cleaned/
├── MASTER_KT_HANDOVER.md / README.md  master handover / pointer
├── main.py / web_app.py              V6 CLI / Flask UI and V6 upload API
├── evaluate_model_readiness.py       historical readiness evaluator
├── requirements.txt                 declared dependencies
├── config/                          preserved shared model configuration
├── configs/                         future V6 training skeleton
├── src/                             inference, event/API infrastructure, training
│   ├── v6_inference.py               shared factory, RGB loading, serialization
│   ├── v6_shadow_pipeline.py / v6_shadow_contract.py
│   ├── v6_shadow_comparison.py / v6_video_input.py
│   ├── pipeline.py
│   ├── segformer_yolo_depthv2_pipeline.py
│   ├── water_region_detector.py / reference_depth_estimator.py
│   ├── dataset.py / train.py / train_water_aware.py
│   └── flood_depth/                  alternate segmentation abstraction
├── scripts/                         data preparation, training, audits, experiments
├── models/                          original/fallback and mask-conditioned weights
│   └── candidate/                   active AND rejected checkpoints
├── training_data/                   images, labels, class folders, new-data candidates
├── training_runs/                   historical splits / grouped manifests
├── evaluation_data/                 evaluation images/manifests
├── reports/                         traces, metrics, audits, freeze registries
├── docs/                            supporting guides / V6 entrypoint architecture
├── templates/                       V6 new-image manifest template, not UI HTML
├── data/                            SQLite / configured quarantine
└── archive/                         legacy CLI / archived training assets
```

Temporary logs/caches and individual images are omitted. Folder name does not determine checkpoint/data status. Ignored or untracked images, reports and weights may not arrive in a clone; obtain the approved artifact bundle separately. `.venv` is machine-local.

## 3. Starting Point / Application Entry Point

The normal recommended image command is:

```powershell
python main.py image "<image_path>" --storage local
```

Active entrypoints are `main.py image`, `main.py video`, `web_app.py` (not webapp.py), `/api/v1/estimate`, `src/worker.py:process_camera_event()`, and the V6 diagnostic image/video scripts. EC2 uses the same commands or event adapter; no separate EC2 predictor exists. Commit `7508638` established the shared boundary; this migration removes legacy execution from normal inference.

```text
main.py image / UI / diagnostic image CLI / video CLI / API / worker / EC2
 → src.v6_inference.load_v6_rgb (PIL uint8 RGB)
 → src.v6_inference.create_v6_pipeline
 → V6ShadowPipeline(EfficientNetDepthSignal()).predict(image_rgb)
 → primary_depth_cm = final_shadow_depth_cm
 → shared v6_depth_payload (primary cm, final cm, numerical owner)
```

The shared loader performs no adapter resizing/normalization. The unchanged model transform remains in `src/efficientnet_depth_signal.py`. None/NaN/infinite depth is unavailable, never zero. Checkpoint load failure raises an explicit error; no random-weight or V5 fallback exists. New API uploads return a synchronous V6 response with HTTP 200. They do not apply legacy reference gates, judge corrections, severity, telemetry writes or automatic temporal aggregation. The old SQLite schema requires confidence/severity fields that V6 does not produce; existing stored telemetry/camera/temporal analytics remain separate historical-data services.

### Legacy Production/Reference Pipeline (V5) application and worker path

This chain describes the former default, before the V6-default migration. It is reference history only. Normal CLI, UI, upload API and worker execution now use shared V6. `/predict` calls shared V6 directly; `FloodApiService.process_camera_upload()` and `src.pipeline.execute_event()` now use shared V6 as well.

```text
image bytes / upload
 → CLI process_image_cli OR Flask route OR worker.process_camera_event
 → FloodApiService.process_camera_upload → FloodEvent
 → src.pipeline.execute_event → get_processor
 → UnifiedEventProcessor.process_event → PIL RGB decode
 → get_segformer_yolo_depthv2_pipeline().predict
 → core depth + ordered snapshots/features
 → camera-window aggregation + FloodIntensityClassifier + FloodResultEvent
 → API reference requirement / optional LLM / final_decision / persistence
```

The audited single-image baseline starts directly at `SegformerYoloDepthV2Pipeline.predict()` through `scripts/audit_full_pipeline_trace.py`, excluding API postprocessing and camera-window state. The processor/core are process-level singletons: restart after configuration/checkpoint changes.

### Code walkthrough of the wrapper dispatch

This historical wrapper selected the staged core using pipeline_mode. Its former application policies are not applied by the current V6 event adapter.

Historical source: `src/pipeline.py` before the default migration (lines 157-166). This excerpt is reference only; the current file is a V6 event adapter.

```python
if self.pipeline_mode == "segformer_yolov8_depthv2_fusion":
    staged = self.segformer_yolo_depth_pipeline.predict(np.array(image))
    depth_cm = float(staged["depth_cm"])
    confidence = float(staged["confidence"])
    method = str(staged.get("method", "segformer_yolov8_depthv2_fusion"))
    action = str(staged.get("action_trigger", "Monitor"))
    metadata["pipeline_trace"] = staged.get("pipeline_trace", [])
    metadata["pipeline_stage_outputs"] = staged.get("pipeline_stage_outputs", [])
    metadata["structured_features"] = staged.get("structured_features", {})
    metadata["visual_cues"] = staged.get("visual_cues", [])
```

## 4. End-to-End Runtime Pipeline

### Current Final Architecture (V6) numerical pipeline

```text
shared uint8 RGB input
 → V6ShadowPipeline.predict()
 → shared EfficientNet primary model produces structured signals
 → EfficientNet primary metric signal
 → primary_depth_cm
 → reporting-only object/relative/semantic/reliability/uncertainty stages
 → final_shadow_depth_cm = primary_depth_cm
```

`NUMERICAL_OWNER` remains `efficientnet_primary_anchor`. Semantics/masks are context only, YOLO/contour signals are diagnostic, and Depth Anything stays relative/context only. Normal inference loads only the shared EfficientNet primary model; it never constructs or executes the Legacy Production/Reference Pipeline (V5). Model architecture, checkpoint loading, transform, maximum-depth scaling and rounding were extracted unchanged into `src/efficientnet_depth_signal.py`; the legacy core inherits those same methods for reference use. Optional semantic/object/relative diagnostics are unavailable from this primary-only source, and reporting flags reflect that absence. Residual Fusion Model (V14) is not a V6 numerical owner. `NoOptionalRefinement` is a no-op; no resolver, guard, correction or averaging replaces the V6 primary depth. Internal classes and fields are not renamed for documentation.

### Legacy Production/Reference Pipeline (V5) extraction and final decision history

The following order is verified from `src/segformer_yolo_depthv2_pipeline.py:predict()`. It is historical/reference material only. Normal V6 inference does not execute this multi-stage path.

```text
RGB validation → classical mask → four semantic signals
 → EfficientNet → conditional muddy-mask replacement
 → object references → Depth Anything relative proxy
 → optional teachers (disabled) → separate contour reference estimate
 → physical features → mask-conditioned candidate
 → calibration/base → EfficientNet correction → model agreement
 → pre-residual guard evaluation (potential caps recorded, not applied)
 → raw V14 / safety policy / applied V14
 → dynamic resolver → mask-conditioned high-flood correction
 → strong-deep correction (disabled) → dry-land guard
 → final no-water guard → dry-road override → final severity/result
```

Methods below are in `SegformerYoloDepthV2Pipeline` unless another class is named. Checkpoint details are in section 7.

| Stage / exact function | Input → output / handoff | Why it exists / model |
|---|---|---|
| `predict` validation | RGB H×W×3 → accepted array/error | Checks dimensions/channels |
| `_segformer_water_mask` → `WaterRegionDetector.detect` | RGB → binary mask and coverage % | Classical HSV/RGB/contrast/muddy-water evidence; **no SegFormer checkpoint** |
| `_no_water_guard_signal`, `_wet_road_no_water_guard_signal`, `_road_scene_classifier_signal`, `_depth_regime_classifier_signal` | RGB → probabilities | Two binary MobileNet guards, four-class MobileNet scene model, EfficientNet regime model; scored now, applied later |
| `_efficientnet_depth_signal` | normalized 224 RGB → cm candidate | Hardneg supervised EfficientNet-B0 |
| muddy fallback inside `predict` | candidate/mask/dry evidence → possible replaced mask | Existing branch: candidate ≥35 cm, coverage <5%, not dry land, no-water not ≥0.92, muddy coverage ≥15% |
| `_yolov8_reference_stage` | RGB + mask → `ReferenceObject` list/backend | Improved legacy `ObjectDetector`, YOLO or contour fallback; count/submersion evidence |
| `_depth_anything_v2_dense_map` | RGB + mask → normalized spatial proxy | HF Depth Anything V2 or synthetic fallback, not metric cm |
| `_depth_teacher_features` | image/mask → teacher diagnostics | `TeacherEnsemble`, currently disabled |
| `ReferenceDepthEstimator.estimate` | RGB → separate reference cm/waterline/cues | Its own HSV/contours, not the YOLO reference objects |
| `_fusion_engine`, `_water_zone_features`, `_estimate_region_depth` | mask/references/proxy/contour estimate → features | Near/mid/far coverage, geometry/count/submersion; legacy proxy p90×120 scale |
| `_mask_conditioned_fusion_depth_signal` | RGB + 24 object + 3 geometry features → cm candidate | EfficientNetV2 fusion; adds candidate to features |
| `_calibration_severity_model` | features → base cm/confidence/action | Rule-based blending/fallbacks/coverage-risk gates; no trained calibration checkpoint |
| `_apply_efficientnet_correction` | base/candidate/evidence → corrected base | Conditional severe-underestimate correction |
| `_record_model_agreement` | candidates/trust → depth/source/review | Hand-weighted agreement/selection |
| `_apply_no_water_guard(..., apply_cap=False)` | base/semantics → potential caps | Keeps numerical depth; preserves capped anchor for V14 |
| `_apply_residual_fusion_model` | checkpoint-normalized feature vector + anchor → raw/applied cm | V14, skip rules, moderate-collapse safety fallback and adjustment bounds |
| `_apply_dynamic_broad_mask_resolver` | residual + regime/physical evidence → resolved cm | Enabled; qualifying agreement can raise depth and request review |
| `_apply_mask_conditioned_high_flood_correction` | current result + candidate/evidence → possible replacement | Enabled independently of `use_for_final_decision` |
| `_apply_strong_deep_flood_correction` | current result → unchanged | Config disabled |
| `_apply_dry_land_guard`, `_apply_no_water_guard` | image/features → zero/capped/unchanged | Final dry/no-water decisions with corroboration |
| dry override inside `predict`, `_depth_to_severity` | guarded result → final cm/severity | Dry probability ≥0.995 can force zero; severity runs last |

Calibration uses `0.65 × reference_depth_cm + 0.35 × dense proxy` when references qualify, followed by caps/gates. This describes current numerical behavior, not proven metric fusion. **V14 is not the final numerical owner.** Ordered `pipeline_stage_outputs` are handoff authority; human-readable `pipeline_trace` contains summaries appended later. API adds another layer (section 26).

### Code walkthrough of the metric candidate

EfficientNet produces a normalized scalar that this loader converts to centimetres using its configured maximum depth. This is a metric candidate in V5 and the sole primary/final metric owner in current V6. The historical signal method yields None for a missing model/transform; the active V6 constructor raises an explicit checkpoint-load error before prediction.

Source: `src/efficientnet_depth_signal.py:_efficientnet_depth_signal()`; extracted unchanged from `src/segformer_yolo_depthv2_pipeline.py` at commit 7508638 (lines 884-890). Enclosing context/imports are omitted.

```python
def _efficientnet_depth_signal(self, image_rgb: np.ndarray) -> Optional[float]:
    if self._efficientnet_model is None or self._efficientnet_transform is None:
        return None
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    tensor = self._efficientnet_transform(image).unsqueeze(0).to(self._efficientnet_device)
    with torch.no_grad():
        return round(float(self._efficientnet_model(tensor).squeeze().item()) * self._efficientnet_max_depth_cm, 2)
```

### Code walkthrough of live residual skip rules

These live rules can leave the incoming depth unchanged even when V14 is loaded. They demonstrate why checkpoint output alone does not reproduce live prediction behavior. This excerpt begins after the earlier no-water risk checks; it is not the full guard policy.

Source: `src/segformer_yolo_depthv2_pipeline.py` (lines 752-761). This is an excerpt from the existing implementation; enclosing context/imports are omitted. Read the linked source before reusing it.

```python
if no_water_guard_high and low_risk_no_water_scene:
    features["residual_fusion_status"] = "skipped_no_water_guard_candidate"
    return depth_cm, confidence, action
skip_low_water = bool(cfg.get("skip_low_water_gate", True))
if skip_low_water and bool(features.get("low_water_gate_applied", False)) and not bool(features.get("shallow_water_gate_exception", False)):
    features["residual_fusion_status"] = "skipped_low_water_gate"
    return depth_cm, confidence, action

if bool(features.get("no_reference_depth_uncertain", False)):
    features["residual_fusion_status"] = "skipped_no_reference_depth_uncertain"
```

## 5. Python File-by-File Explanation

### Current Final Architecture (V6) files

| File | Responsibility |
|---|---|
| `src/v6_inference.py` | Shared factory, PIL RGB loader and primary/final cm serialization |
| `src/efficientnet_depth_signal.py` | Unchanged primary model loader/transform/scaling, without legacy pipeline execution |
| `src/v6_shadow_pipeline.py` | Single V6 prediction implementation, primary/final ownership and reporting-only stages |
| `src/v6_shadow_contract.py`, `src/v6_shadow_comparison.py` | Units, signal authority, immutable result/comparison contracts |
| `web_app.py` | Flask image UI, V6 `/predict` and V6 upload API; historical-data analytics remain separate |
| `scripts/run_v6_shadow_comparison.py` | Optional diagnostic image/hash verification and stage reporting through shared V6; legacy comparator fields are null |
| `src/v6_video_input.py`, `scripts/run_v6_shadow_video.py` | Decode, save JPEG/reload RGB, shared prediction, CSV/run summary |
| `tests/test_v6_entrypoint_parity.py`, `tests/test_v6_web_ui.py` | Adapter RGB/output parity and UI contract checks |

### Active application adapters and historical reference dependencies

| File | Purpose / important classes/functions | Called by → dependencies | Inputs → outputs |
|---|---|---|---|
| `main.py` | `main`, `process_image_cli`, `process_video_cli`, `summarize_event_result`, `write_final_prediction_record` | User → shared V6, local/S3 I/O | V6 cm console and saved-JPEG video CSV; old summary helpers are unused history |
| `web_app.py` | `index`, `predict`, `get_v6_pipeline`, upload API | Flask → shared V6 directly or through the V6 event adapter | original image bytes → V6 final cm; inline HTML/CSS/JS |
| `src/pipeline.py` | `UnifiedEventProcessor.process_event`, `execute_event`, `get_processor` | API/worker → shared V6, retry/observability | shared RGB → V6EventResult with primary/final cm; no averaging |
| `src/segformer_yolo_depthv2_pipeline.py` | `predict`, loaders/fusion/guards/resolver; `ResidualFusionDepthModel`, `DepthRegimeHead`, `MaskConditionedFusionDepthModel`, `ReferenceObject` | wrapper/audits → torch/torchvision/timm/cv2/transformers/reference modules | RGB → cm/trace/features |
| `src/water_region_detector.py` | `WaterRegionDetector.detect`, `looks_like_dry_land`, `get_water_bounding_boxes` | core → cv2/NumPy | RGB → mask/coverage/boxes/dry evidence; training wrappers also here |
| `src/reference_depth_estimator.py` | `ReferenceDepthEstimator.estimate`, `detect_reference_objects`, vehicle/person methods, `label_from_image` | core/label utility → own mask/contours | RGB → heuristic cm/waterline/confidence/guide |
| `archive/legacy_cli/modules/object_detection.py` | `ObjectDetector` | core optional import | RGB/mask → improved references; **archived location remains live dependency** |
| `src/settings.py` | `RuntimeSettings`, `AppConfig`, `load_settings_dict` | runtime → YAML/Pydantic/env | raw config/overlays/secrets → dictionary |
| `src/event_contract.py` | `FloodEvent`, `FloodResultEvent`, `FloodFailureEvent` | API/wrapper/worker | base64/location/schema → validated event/result/error |
| `src/api_service.py` | `FloodApiService.process_camera_upload`, historical-data methods | Flask/worker → V6 event adapter; repository for history | upload → V6 result; no judge/reference gate/telemetry write |
| `src/aggregator.py` | `SlidingWindowAggregator.push`, `window_state`, `SensorPayload` | historical reference only → Redis/memory | camera cm/confidence/time → aggregate metres/action |
| `src/geospatial_classifier.py` | `FloodIntensityClassifier.classify`, GeoJSON helpers | historical reference only → thresholds | cm → severity/label/color/action |
| `src/storage.py` | `FloodRepository`, `TelemetryRecord` | API/temporal → sqlite3 | records → SQLite persistence |
| `src/temporal_analysis.py` | `TemporalFloodAnalyzer.create_temporal_sequence` | API/worker → telemetry | recent rows → temporal sequence; not image checkpoint |
| `src/llm_judge.py` | `LLMJudge.judge`, response parsers | API → Google API/key | image/prediction → optional review/correction; outside core parity |
| `src/worker.py` | `process_camera_event`, `analyze_temporal_sequence`, optional Celery tasks | queue → API service | task payload → response |
| `src/middleware/retry.py`, `src/middleware/observability.py`, `src/dlq.py` | retry/logging/dead-letter infrastructure | event infrastructure | errors/events → retry/log/failure records |

### Training, evaluation and non-active modules

| File | Role / logic / input-output contract |
|---|---|
| `src/dataset.py` | Training `FloodDataset`, `S3DataHandler`, `create_dataloaders`: images/manifest → tensors; strict labels, checksum/corruption checks, quarantine. Generic depth casts `int(float(...))`, normalizes at 100 cm; not candidate trainer's float/180 contract. |
| `src/train.py` | Training `build_model`, `EarlyStopping`, generic utilities used by water-aware trainer. |
| `src/train_water_aware.py` | Training `WaterAwareTrainer`; config/dataloaders → original checkpoint; imports region-aware wrappers from water detector. |
| `scripts/train_candidate_depth_model.py` | Training `FloodDepthDataset`, `build_model`: image/float depth → model; clips/normalizes targets to `--max-depth-cm`. |
| `scripts/train_no_water_guard.py` | Training binary MobileNet, class folders → checkpoint/class metadata; random stratification is not automatic group-clean validation. |
| `scripts/train_road_scene_classifier.py` | Training `RoadSceneDataset`, `build_model`, `evaluate`: fixed scene splits → four-class checkpoint. |
| `scripts/train_depth_regime_classifier.py`, `scripts/train_depth_regime_classifier_finetune.py` | Training `RegimeHead`/`FineTunedRegimeModel`: labeled splits → EfficientNet embeddings/head/checkpoint/reports. |
| `scripts/train_fusion_depth_model.py` | `extract_feature_row`, `build_feature_cache`, `load_or_build_features`, `FusionDepthModel`: pipeline signals/splits → provenance cache or direct regressor. |
| `scripts/train_residual_fusion_depth_model.py` | `ResidualFusionDepthModel`, `make_tensors`, `sample_weights`, `train`: normalized features/base → residual checkpoint/historical test report. |
| `scripts/train_residual_fusion_depth_model_dev.py` | Development preparation despite name: checks augmentation roles/writes integrity; does not train; historical 24-image assumptions. |
| `scripts/run_residual_fusion_depth_dev.py` | Development extraction/training/parity runner, train/val and clean-only/dry-run controls; historical candidates rejected. |
| `scripts/audit_full_pipeline_trace.py` | Audit `read_rows`, `stage_summary`, `main`: approved filenames/manifest → JSON snapshots/features; blocks consumed-test/challenge terms. |
| `scripts/evaluate_live_pipeline.py` | Audit split images/labels → core CSV; default historical test, explicitly choose new val; no API postprocessing. |
| `scripts/evaluate_generalization_manifest.py` | Audit `active_artifacts`, `metric_summary`: manifest → resumable core evaluation/fingerprints. |
| `evaluate_model_readiness.py` | Audit dataset/manifest, bands/contradictions/gates → timestamped reports; historical evaluator, not automatic V5 certification. |
| `scripts/evaluate_candidate_depth_model.py`, `scripts/compare_mask_conditioned_fusion.py` | Isolated comparison, not full-live acceptance. |
| `scripts/audit_dataset_splits.py` | Exact SHA/perceptual duplicates/filename families/missing images across materialized splits; cannot automatically fix leakage or infer sessions. |
| `scripts/prepare_training_and_splits.py`, `scripts/prepare_dataset.py`, `scripts/validate_training_labels.py`, `scripts/auto_fill_labels.py` | Historical preparation/label tools; simple splits/inferred labels do not establish measured truth/group cleanliness. |
| `scripts/prepare_v15_foundation.py` | Historical `UnionFind` grouping/manifests, not inference logic. |
| `scripts/prepare_clean_classifier_split.py`, `scripts/audit_new_image_batch_duplicates.py` | Useful development data controls, not checkpoint promotion. |
| `src/depth_teachers.py` | Optional `TeacherEnsemble`, disabled. |
| `src/flood_depth/segmentation_engine.py` | Alternate lazy legacy/DeepLab abstraction, not called by core; `SEGMENTATION_BACKEND` does not switch V5 masks. |
| `src/v15_residual_spec.py`, `src/two_stage_models.py`, `src/two_stage_residual_spec.py` | Experimental architectures/specifications, not active Residual Fusion Model (V14). |
| `scripts/train_scene_guard.py`, `scripts/train_flood_depth_expert.py`, `scripts/train_residual_regime_router.py`, `scripts/evaluate_hierarchical_depth_router.py` | Older/experimental guard/expert/router paths, not active chain. |
| `scripts/run_phase4_shallow_meaningful_router.py`, `scripts/run_phase4_image_shallow_meaningful_router.py` | Rejected tabular/pixel shadows. |
| `scripts/train_v15_residual_candidate.py`, `scripts/run_two_stage_development_experiment.py`, `scripts/evaluate_frozen_collapse_safety.py` | Rejected work; retain evidence, do not enable. |
| `archive/legacy_main.py`, other legacy modules | Old severity/video/water CLI, not V5; `mc_dropout.py` supports fallback confidence, not staged core. |

## 6. Configuration

Preserved [config/config.yaml](config/config.yaml) is loaded by `load_settings_dict()`. Current Final Architecture (V6) uses the existing configured models through the shared factory; the correction/guard/resolver effects in the following table describe Legacy Production/Reference Pipeline (V5) only and do not own V6 centimetres. `FLOOD_CONFIG_PATH`/`FLOOD_APP_ENV` select path/environment; default production. Dotted overlays merge; `${SECRET:ENV_VAR_NAME}` requires that variable. Record resolved config as well as raw hash.

| `inference` section | Current effect |
|---|---|
| `pipeline_mode` | `segformer_yolov8_depthv2_fusion`; other modes use wrapper single-frame model |
| `model_path` | Original EfficientNet fallback, not primary core candidate |
| `depth_model` | Enabled HF small Depth Anything, CPU device -1; failure activates proxy |
| `efficientnet_signal` | Enabled hardneg/180 cm/CPU, corrections on |
| `mask_conditioned_fusion_signal` | Enabled CPU; `use_for_final_decision: false` does not disable enabled high-flood correction. Nominal trigger mask≥75/current≤70/gap≥25 cm, coverage≥45/near-or-mid≥35%, plus corroboration branches |
| `no_water_guard` | Primary 0.99, coverage/near≤5%; wet guard 0.995, coverage≤12/near≤8%; actual zeroing also needs corroboration |
| `road_scene_classifier` | Enabled, `advisory_only: false`; dry override0.995; wet cap enabled0.995→3 cm, coverage/near≤12/8%; shallow cap disabled |
| `dynamic_broad_mask_resolver` | Enabled/classifier loaded; broad-mask/strong-reference/collapse recovery, scaled signals/IQR/evidence; min confidence0.35, max blend0.85, automatic confidence0.70 |
| `residual_fusion_signal` | Residual Fusion Model (V14) enabled/corrections/low-water skip; live adjustment120 cm, aligned exception enabled/alignment15 cm/review delta20 cm |
| `moderate_flood_safety_fallback` | Enabled existing V5 protection: collapse≤0.5, base/mask≥20, spread≤8 cm and semantic limits; distinct from rejected collapse-only experiment |
| `strong_deep_flood_correction`, `depth_teachers` | Disabled |
| MC dropout settings | Alternate single-frame wrapper only, not core confidence |
| `llm_judge` | Config enabled/corrections permitted; actual API use depends on key |

`training` controls historical defaults; `data` labels/local/S3/filter/quarantine; `aggregator` windows/actions; `storage` SQLite; `event_processing` retry/DLQ. AWS/ECS/Lambda/LitServe fields are intentions, not deployment proof. Some fields are unused/misleading. Keep all thresholds frozen for reproduction; this handover recommends no threshold changes.

## 7. Current Primary Checkpoint and Historical Model Registry

The eight registered local artifacts are preserved Legacy Production/Reference Pipeline (V5)/shared-extraction dependencies (section 27 records the historical hash audit). Only the hardneg EfficientNet checkpoint owns Current Final Architecture (V6) metric depth. The current parity smoke verified its SHA-256 and eval-mode loading; other model roles below describe the historical V5 extraction/comparison path.

| Model | Checkpoint | Purpose / input-output | Called from | Historical V5 use? |
|---|---|---|---|---|
| Original EfficientNet-B0 | `models/best_flood_model_water_aware.pth` | 224 RGB→normalized depth | wrapper `_load_weights`, alternate mode | Configured fallback, not primary core |
| Hardneg EfficientNet-B0,256/128 sigmoid head | `models/candidate/best_flood_model_water_aware_hardneg.pth` | 224 / ImageNet RGB→sigmoid×180 cm | EfficientNet loader/signal | Yes |
| Mask-conditioned `efficientnetv2_rw_s` | `models/FloodDepth-MaskConditionedFusion.pth` | 384 RGB+24 object+3 geometry→log-depth/inverse log1p cm, ordinal head | mask loader/signal/correction | Yes |
| Binary MobileNetV3-Small | `models/candidate/no_water_guard_teammate_augmented.pth` | 224 RGB→no_water/water | primary guard loader/signal | Yes |
| Wet-road binary MobileNetV3-Small | `models/candidate/wet_road_no_water_guard_test0916.pth` | same two-class contract | wet guard loader/signal | Yes |
| Four-class MobileNetV3-Small | `models/candidate/road_scene_classifier_4class_shallow_v2.pth` | 224 RGB→dry/wet/shallow/meaningful | scene loader/signal/caps/residual | Yes |
| EfficientNet-B0+`DepthRegimeHead` | `models/candidate/depth_regime_classifier_v6_targeted_conservative.pt` | resize 256 / center-crop 224→regime probabilities | regime loader/signal/resolver | Yes |
| Residual Fusion Model (V14) MLP 40→48→24→1,tanh | `models/candidate/residual_fusion_depth_model_ankle_v14_no_leak.pt` | standardized features+base→0–180 cm | residual loader/application | Yes |
| YOLOv8n | `yolov8n.pt` | image→object references | detector/reference stage | Conditional on load |
| HF Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` (remote ID, not repo filename) | image→relative map | transformers loader | Configured; verify backend |

V14 checkpoint supplies authoritative feature names/order/mean/std, anchor and residual bound. Python defaults are not enough for arbitrary weights. Mask fusion concatenates projected visual/object/geometry embeddings, not measured-depth pixels.

## 8. Important Signal Contracts

Current Final Architecture (V6) authority: EfficientNet = `PRIMARY_METRIC`; semantics/masks and relative depth = `CONTEXT_ONLY`; object/contour proxies = `DIAGNOSTIC_ONLY`; engineered region and mask-conditioned candidates = advisory only. `primary_depth_cm` and `final_shadow_depth_cm` are never inferred from Legacy Production/Reference Pipeline (V5) metre fields. The following table preserves the previous V5 audit findings and numerical uses.

| Signal | Contract / audit finding |
|---|---|
| EfficientNet candidate | Supervised cm, strongest standalone cm signal in approved clean analysis; shallow overestimation persists |
| Water mask | Binary evidence/percentages/geometry, not depth |
| YOLO references | Boxes/classes/confidence/count/submersion context; Phase 1 per-object height/depth proxies diagnostic-only, `used_by_current_reference_fusion: false` |
| Contour reference cm | Separate heuristic still numerically used by V5; unreliable direct cm. YOLO count does not substantiate it |
| Depth Anything | Per-image normalized relative proxy. Legacy p90×120 retained for parity, **not metric cm** |
| Region depth | Engineered geometry/proxy cm scale; advisory reliability despite active numerical use |
| Mask-conditioned candidate | Learned cm, checkpoint validation R²0.297 per audit; only eight Phase 1 diagnostic traces; high-flood correction nevertheless enabled |
| Semantic probabilities | Context, not cm; repeatedly affects residual/caps/zeroing/resolver |
| Residual Fusion Model (V14) | Correction around checkpoint base; raw differs from applied due to skip/safety/bounds; later rules replace it |

See [full audit](reports/full_pipeline_audit_20261002/full_pipeline_audit_report.md), [cleanup](reports/full_pipeline_audit_20261002/phase1_contract_cleanup_report.md), [Phase 2 reliability](reports/full_pipeline_audit_20261002/phase2_signal_reliability_report.md). Phase 2 saved-table analysis used 465 TRAIN/99 VALIDATION: EfficientNet validation MAE 9.17 cm versus contour 81.26 cm. Recommended metric-source changes were **not implemented** in frozen V5.

## 9. Data and Dataset Structure

Historical image pools include `training_data/images`, the former `training_data/un_labled_images` (historical spelling retained; folder absent in the current workspace), `training_data/water_presence`, scene/depth folders, `evaluation_data/images`, and materialized `training_runs/.../images/{train,val,test}`. `training_data/labels.csv`, `labels_aligned.csv`, `labels_clean.csv` are historical label sources, not interchangeable authorities. Current local data additions/deletions and changed evaluation manifests must be preserved.

Basic label CSV: `filename,depth_cm`, optionally SHA-256. Candidate trainer keeps floating centimetres; generic `FloodDataset` currently truncates them to integers. Grouped proposal manifests add `image_id,image_path,depth_bucket,frozen_group_id,grouping_confidence,source_family`.

| Artifact | Current role |
|---|---|
| `training_runs/v15_clean_grouped_proposal/labels_train.csv` | Approved historical development TRAIN, 465 rows in Phase 2 |
| same directory `labels_val.csv` | Frozen historical development VALIDATION, 99 rows; extensively analyzed, not a new final set |
| same directory `labels_test.csv` | **Consumed internal test; reporting-only, no further tuning/selection/retraining/architecture decisions** |
| `reports/external_60_data_audit_20260930/external_challenge_frozen_manifest.csv` | **Consumed 12-image external challenge; reporting-only, no future tuning** |
| `reports/external_60_data_audit_20260930/external_57_development_manifest.csv` | Historical role-audited development batch; only approved role/confidence/exposure rows are eligible |
| `evaluation_data/evaluation_manifest.csv` | Locally modified general evaluation source; not an untouched-set certificate |
| unlabeled and `training_data/candidate_new_data` pools | Review-only until validated, deduplicated, grouped and independently labeled |
| `data/quarantine` | Configured corrupt/checksum/missing-label quarantine; do not automatically recycle |

Manifest membership is authoritative: approved validation rows can physically point to historical `images/train` or `images/test`. Some groups have `uncertain_singleton_no_link_evidence`; absence of a known link is not proof of independence. See [freeze ledger](reports/architecture_investigation_freeze_20261002.json).

Keep every source/session/video/burst/location family in one split. Adjacent frames share viewpoint, background, weather and objects, permitting memorization and inflated validation performance. Use exact hashes, perceptual similarity and human session links together; filename families alone are insufficient.

## 10. Training Architecture

Separately trainable: supervised image-depth regression, binary guards, scene classifier, depth-regime classifier and feature residual fusion. Classical masks/calibration/resolver rules are not trainable checkpoints. No verified project-specific YOLO/Depth Anything retraining workflow is established. Full mask-conditioned training source/provenance is incomplete.

| Component | Trainer / input / important controls | Output / evidence |
|---|---|---|
| Original water-aware depth | `src/train_water_aware.py`, YAML/generic dataloaders, region-aware wrappers | Original checkpoint; complete accepted run history unavailable |
| EfficientNet candidate | `scripts/train_candidate_depth_model.py`; train/val image folders/CSV/base checkpoint; max-depth, epochs, batch, LR, seed, freeze-backbone | Explicit `--output`; isolated candidate evaluator |
| Binary guards | `scripts/train_no_water_guard.py`; no_water/water folders, val fraction, seed/device/pretrained | Checkpoint/class metadata; exact accepted runs incomplete |
| Scene classifier | `scripts/train_road_scene_classifier.py`; fixed scene splits, epochs/LR/device | Checkpoint; exact shallow-v2 invocation not reconstructed |
| Regime classifier | `scripts/train_depth_regime_classifier_finetune.py`; depth split, warm start, unfreeze/class penalties | Model/eval/summary; v6 CSV/JSON retained in reports |
| Residual Fusion Model (V14) family | `scripts/train_residual_fusion_depth_model.py`; feature cache/split, anchor, max-residual, weights, seed | Checkpoint; `reports/fusion_ankle_depth_v14_no_leak_features.csv`, `reports/residual_fusion_ankle_v14_no_leak_eval.csv` |

EfficientNet learns clipped normalized targets with sigmoid output. V14 standardizes features using TRAIN statistics, adds a tanh residual to its selected anchor and clamps to 0–180 cm. Its checkpoint saves architecture/state, feature names/mean/std, depth/residual bounds, base feature, validation MAE, `training_config`, and meaningful-flood-floor metadata.

Historical validation selection is not independent final evidence. Future hyperparameter/calibration selection should use group-locked TRAIN CV/OOF, frozen before validation review. V2–V5 “residual_fusion” filenames used direct `FusionDepthModel` architecture: not compatible merely because the basename says residual. See [residual notes](RESIDUAL_FUSION_TRAINING.md).

### Code walkthrough of the image training target

The image trainer decodes RGB and clips and normalizes the target using max_depth_cm. Keep this scaling aligned with candidate inference; the separately returned depth_cm retains the original label.

Source: `scripts/train_candidate_depth_model.py` (lines 47-56). This is an excerpt from the existing implementation; enclosing context/imports are omitted. Read the linked source before reusing it.

```python

def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    image_path, depth_cm = self.items[idx]
    image = Image.open(image_path).convert("RGB")
    target = np.clip(depth_cm, 0.0, self.max_depth_cm) / self.max_depth_cm
    return {
        "image": self.transform(image),
        "target": torch.tensor(target, dtype=torch.float32),
        "depth_cm": torch.tensor(depth_cm, dtype=torch.float32),
    }
```

### Code walkthrough of the EfficientNet training head

This trainable architecture is an EfficientNet-B0 backbone with a regression head ending in sigmoid. weights=None means this function itself does not load pretrained weights; checkpoint loading is handled separately by load_matching_checkpoint.

Source: `scripts/train_candidate_depth_model.py` (lines 59-72). This is an excerpt from the existing implementation; enclosing context/imports are omitted. Read the linked source before reusing it.

```python
def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    return model
```

### Code walkthrough of the bounded residual

The residual network predicts a bounded correction around the supplied base candidate. This trainer excerpt explains the arithmetic; the live pipeline adds skip rules, confidence checks and subsequent overrides, so raw V14 and applied V14 must be inspected separately.

Source: `scripts/train_residual_fusion_depth_model.py` (lines 63-65). This is an excerpt from the existing implementation; enclosing context/imports are omitted. Read the linked source before reusing it.

```python
def forward(self, x: torch.Tensor, base_depth_cm: torch.Tensor) -> torch.Tensor:
    residual_cm = self.net(x) * self.max_residual_cm
    return torch.clamp(base_depth_cm + residual_cm, min=0.0, max=180.0)
```

### Code walkthrough of residual feature normalization

Use feature order from FEATURE_NAMES and compute normalization on TRAIN only. Pass the saved TRAIN mean and std when preparing validation features; recomputing them on validation changes the contract and introduces evaluation contamination.

Source: `scripts/train_residual_fusion_depth_model.py` (lines 88-100). This is an excerpt from the existing implementation; enclosing context/imports are omitted. Read the linked source before reusing it.

```python
if mean is None:
    mean = x.mean(axis=0)
if std is None:
    std = x.std(axis=0)
std = np.where(std < 1e-6, 1.0, std)
x = (x - mean) / std
return (
    torch.tensor(x, dtype=torch.float32),
    torch.tensor(base, dtype=torch.float32),
    torch.tensor(y, dtype=torch.float32),
    mean,
    std,
)
```

## 11. How to Retrain / Fine-Tune in Future

1. Create a separate experimental branch; preserve Legacy Production/Reference Pipeline (V5) config/checkpoints/hashes.
2. Collect genuinely new images with scene/depth/provenance labels.
3. Validate decoding/labels; ambiguous depth stays review-only.
4. Audit exact/near duplicates against prior pools, mark exposure, group source/session/location families.
5. Build new group-clean TRAIN and frozen VALIDATION. Freeze new untouched internal test and external challenge before training; restrict access.
6. Train only on TRAIN. Use group-locked TRAIN CV/OOF for selection/calibration/correction bounds.
7. Freeze candidate and evaluate VALIDATION overall/bucket/semantic/catastrophic metrics; retain review/missing rows in accounting.
8. Verify preprocessing/features/units/backends/source/config/model/data hashes and full-live parity. Rebuild caches when extractors change.
9. Freeze architecture, thresholds and config; use the **new** internal final set once and external challenge once. Never tune from those results.
10. After acceptance register provenance/hashes, update experimental paths, regenerate readiness evidence and deploy with rollback. Never overwrite V5.

Accepted component paths: `efficientnet_signal.model_path` plus verified scale; binary guard paths; road-scene path; resolver `classifier_model_path`; residual path with matching features; mask fusion path with validated 24/3 contract. Changed extraction signals invalidate dependent residual/caches. A new path or lower residual-only MAE is not acceptance.

Verified EfficientNet interface template (placeholders are new approved inputs/outputs, not existing filenames or an accepted run recipe):

```powershell
python -m scripts.train_candidate_depth_model --train-images <new_train_images> --train-labels <new_train_labels_csv> --val-images <new_validation_images> --val-labels <new_validation_labels_csv> --base-model models/candidate/best_flood_model_water_aware_hardneg.pth --output <new_candidate_checkpoint> --max-depth-cm 180 --device cpu
python -m scripts.evaluate_candidate_depth_model --images-dir <new_validation_images> --labels <new_validation_labels_csv> --production-model models/candidate/best_flood_model_water_aware_hardneg.pth --candidate-model <new_candidate_checkpoint> --production-max-depth-cm 180 --candidate-max-depth-cm 180 --out <new_validation_comparison_csv>
```

All flags are verified in the parsers; unspecified hyperparameters use script defaults, not a recommended optimum. The trainer selects using validation each epoch and loads only shape-matching base tensors: inspect tensor-match counts and record selection exposure. A strict TRAIN-CV selection policy requires an explicitly adapted training workflow. Do not run `src/train_water_aware.py` unchanged for a future experiment: it writes the original active/fallback path.

For a residual candidate, choose the checkpoint-compatible anchor (`pre_residual_output_cap_capped_depth_cm` for Residual Fusion Model (V14)), feature normalization/order and residual bound, then use a train/validation-only workflow. The historical trainer is not a safe turnkey recipe for new untouched final sets. The development runner also hardcodes 465/99 clean rows and 24 augmentation rows: it needs a reviewed adaptation for new manifests. No generic safe future-retraining command exists for arbitrary new data yet; preserve that limitation rather than silently weakening data controls.

## 12. Training Scripts Reference

Examples are source-verified parser interfaces, **not executed during this handover**. Help commands avoid inventing accepted training hyperparameters. Run repository-importing scripts as `python -m scripts...` from root. New-data templates are in section 15.

| Script | Trains/Evaluates | Required inputs | Output | Example command |
|---|---|---|---|---|
| `src/train_water_aware.py` | Original depth | YAML/generic data | Water-aware weights | `python src/train_water_aware.py --help` |
| `scripts/train_candidate_depth_model.py` | EfficientNet | Train/val images/labels/base | `--output` checkpoint | `python -m scripts.train_candidate_depth_model --help` |
| `scripts/evaluate_candidate_depth_model.py` | Depth comparison | Images/labels/two models | CSV | `python -m scripts.evaluate_candidate_depth_model --help` |
| `scripts/train_no_water_guard.py` | Binary guard | Class folders | Checkpoint | `python -m scripts.train_no_water_guard --help` |
| `scripts/train_road_scene_classifier.py` | Four-class scene | Fixed scene splits | Checkpoint | `python -m scripts.train_road_scene_classifier --help` |
| `scripts/train_depth_regime_classifier.py` | Regime head | Labeled depth splits | Checkpoint/CSV/JSON | `python -m scripts.train_depth_regime_classifier --help` |
| `scripts/train_depth_regime_classifier_finetune.py` | Regime fine-tune | Split/optional warm start | Checkpoint/CSV/JSON | `python -m scripts.train_depth_regime_classifier_finetune --help` |
| `scripts/train_fusion_depth_model.py` | Features/direct fusion | Train/val/test split | Cache/model/report | `python -m scripts.train_fusion_depth_model --help` |
| `scripts/train_residual_fusion_depth_model.py` | Historical residual | Cache/split/anchor | Model/report | `python -m scripts.train_residual_fusion_depth_model --help` |
| `scripts/run_residual_fusion_depth_dev.py` | Development residual | Approved train/val/role data | Features/model/metrics | `python -m scripts.run_residual_fusion_depth_dev --help` |
| `scripts/train_residual_fusion_depth_model_dev.py` | Augmentation preparation | Historical role manifest | Approval/integrity | `python -m scripts.train_residual_fusion_depth_model_dev --help` |
| `scripts/audit_dataset_splits.py` | Duplicates/families | Materialized splits | JSON | `python -m scripts.audit_dataset_splits --help` |
| `scripts/evaluate_live_pipeline.py` | Live core | Split/explicit val | CSV | `python -m scripts.evaluate_live_pipeline --help` |
| `scripts/evaluate_generalization_manifest.py` | Manifest live eval | Approved manifest | Reports/hashes | `python -m scripts.evaluate_generalization_manifest --help` |
| `scripts/audit_full_pipeline_trace.py` | Trace | Approved manifest/names | JSON | Section 14 |
| `evaluate_model_readiness.py` | Historical gates | Images/labels | Timestamped reports | `python evaluate_model_readiness.py --help` |

Rejected/obsolete families: V15, two-stage, tabular/pixel routers, collapse-only safety, hierarchical residual routers/experts, older scene guards. Reports are evidence, not deployment instructions. Existing dev scripts may contain hardcoded row counts/defaults. **Full-cache/historical residual trainers enumerate train/val/test and may automatically report test:** do not point them at consumed sets or expose a new final set during development. A new safe workflow must limit development to train/validation.

## 13. How to Run Prediction

### Current Final Architecture (V6) commands

From repository root with an activated, verified environment and configured model bundle:

```powershell
python main.py image "<image_path>" --storage local
python main.py web --host 127.0.0.1 --port 5000
# Browser: http://localhost:5000
python main.py video "<video_path>" --storage local --output video_analysis.csv --skip-frames 1
# Optional bounded video utility:
python -m scripts.run_v6_shadow_video --video "<video_path>" --output-dir "<run_directory>" --max-frames 1 --skip-frames 1
```

The normal image command prints **Flood Depth Estimator – V6** and **Estimated Flood Depth: XX.XX cm**, followed by shared primary/final cm and numerical-owner JSON. It no longer writes the historical final-prediction CSV/Parquet. The storage default remains AWS for compatibility; explicitly select local for local files. AWS changes input retrieval only. Camera/location options remain accepted; image CLI does not ingest camera events.

The UI uploads original bytes to `/predict` and displays final cm directly. `/api/v1/estimate` uses the same shared V6 through the event adapter, returning HTTP 200 with `result.primary_depth_cm` and `result.final_shadow_depth_cm`. Coordinates remain required by the existing event input contract. No normal path uses legacy metre output, fusion, guards, corrections, averaging or fallback.

`scripts.run_v6_shadow_comparison` is **optional diagnostic/stage reporting**, not required for normal inference. Historical V5 comparator/difference fields are null because the default factory no longer executes V5:

```powershell
python -m scripts.run_v6_shadow_comparison --image "<approved_image>" --expected-sha256 <approved_sha256> --output "<diagnostic_json>"
```

Video uses the saved-JPEG/reload-RGB flow. `main.py video` uses `--output`; the bounded utility uses `--output-dir` and writes predictions.csv/run_summary.json. Invalid images and model failures are explicit errors. None/NaN/infinite depth is unavailable, never zero; 0.0 is a valid finite prediction.

The earlier uv launcher failure is an environment issue, not a model mismatch. The 2026-10-08 real parity smoke used installed CPython 3.14 with existing environment packages. Record the actual working interpreter/package set before EC2 deployment. Legacy core/audit scripts and archived commands remain reference utilities only.

## 14. How to Debug One Wrong Prediction

For Current Final Architecture (V6), first verify original file hash, shared PIL RGB input, loaded EfficientNet checkpoint/backend, and V6 primary/final stage snapshots using the approved-image V6 CLI. A mismatch between entrypoints requires investigating bytes, JPEG encoding, RGB/preprocessing, environment and loaded model identity; do not repair it with UI corrections or model changes. The following detailed trace workflow is retained for Legacy Production/Reference Pipeline (V5) reference/extraction investigation.

Verified one-image command, not run for this documentation:

```powershell
python scripts/audit_full_pipeline_trace.py --manifest training_runs/v15_clean_grouped_proposal/labels_val.csv --filenames image_4.jpg --output reports/kt_one_image_trace.json
```

Use explicit approved TRAIN/VALIDATION membership, never consumed test/challenge. Empty filename lists can select all rows, so keep names explicit. Output includes `pipeline_trace`, ordered `pipeline_stage_outputs`, all `features`, actual manifest depth and `stage_summary`.

Find the first erroneous handoff:

1. Check label/source/orientation/colors and actual segmentation backend/coverage. JSON contains mask statistics, not a saved mask image; visualize `WaterRegionDetector.detect()` separately if necessary.
2. Inspect no-water/wet/scene/regime probabilities, class order and availability. Wrong semantics can affect multiple later stages.
3. Compare `efficientnet_candidate_depth_cm` to independent label. Correct candidate may be damaged later.
4. Inspect objects/confidence/area/submersion/diagnostic dispersion versus the **separate** contour estimator method/confidence/reference cm.
5. Inspect relative backend/range, not as cm; inspect region/mask candidates.
6. Follow `calibration_base`, `efficientnet_correction`, `model_agreement` and selected source.
7. Compare pre-residual input with recorded capped anchor. Inspect `residual_feature_contract_diagnostics`, raw `residual_fusion_depth_cm`, applied `residual_fusion_applied_depth_cm`, delta/status/safety/skip.
8. Follow resolver/high-flood correction/disabled strong-deep/dry-land/final no-water/dry override. Record first stage causing wrong final value.
9. For API differences check metre conversion, camera state, `decision_source`, judge and reference-skip policy.

Saved [Phase 1 example](reports/full_pipeline_audit_20261002/phase1_final_trace_example.json): labelled 35 cm image, base 22.95 → applied Residual Fusion Model (V14) 0 → resolver/guards remain 0. This is a residual-path failure in that core trace, not proof of dry ground. Record filename/session/label confidence, first failing stage, snapshots and taxonomy.

### Code walkthrough of one image trace capture

The trace script decodes an approved manifest image to RGB, calls the core directly, and records stage snapshots alongside the manifest label. Inspect the earliest incorrect stage, rather than judging only the final number. The enclosing loop validates image availability and manifest membership.

Source: `scripts/audit_full_pipeline_trace.py` (lines 82-96). This is an excerpt from the existing implementation; enclosing context/imports are omitted. Read the linked source before reusing it.

```python
image = np.asarray(Image.open(image_path).convert("RGB"))
result = pipeline.predict(image)
features = result.get("structured_features", {}) or {}
output.append(
    {
        "split_manifest": str(manifest),
        "filename": row["filename"],
        "image_path": row["image_path"],
        "actual_depth_cm": float(row["depth_cm"]),
        "pipeline_trace": result.get("pipeline_trace", []),
        "pipeline_stage_outputs": result.get("pipeline_stage_outputs", []),
        "features": features,
        "stage_summary": stage_summary(features, result),
    }
)
```

## 15. Model Readiness Process

[Legacy Production/Reference Pipeline (V5) manifest](reports/flood_model_readiness_guard_v5_manifest.json) records branch/base `874a6b1`, raw config/source/model hashes, exclusions and parity scope. A dedicated one-command V5 manifest generator was not identified: this is a recorded freeze ledger.

Current artifacts came from static source/checkpoint/preprocessing inspection, eight approved train/validation live traces before/after Phase 1, and saved feature analysis. See [full audit](reports/full_pipeline_audit_20261002/full_pipeline_audit_report.md), [cleanup](reports/full_pipeline_audit_20261002/phase1_contract_cleanup_report.md), [parity](reports/full_pipeline_audit_20261002/phase1_parity_summary.json), [Phase 2](reports/full_pipeline_audit_20261002/phase2_signal_reliability_report.md), [split metrics](reports/full_pipeline_audit_20261002/phase2_train_validation_split_metrics.json).

`evaluate_model_readiness.py` invokes the core pipeline directly, not API postprocessing. Its manifest requires `image_path` for matching (rows without it are skipped), with `expected_flood`, `expected_depth_cm`, `scene_type`, `label_status` and optional notes; missing flood labels can fall back to filename inference. Supply independent labels explicitly for valid readiness evidence. It searches its supported image extensions under `--input-dir`; a generic training `depth_cm` column alone is not the same label contract. Outputs are timestamped readiness JSON under `reports`; no `--output` option or CSV writer is defined.

For a future candidate:

1. Register commit/dirty state, source/raw+resolved config/model/manifest hashes and packages/devices/HF cache revision.
2. Verify actual segmentation/reference/dense/classifier/residual backends; fallback changes the evaluated system.
3. Trace representative approved train/validation rows; validate order/final versus last snapshot/raw-applied distinction/later overrides.
4. Check missing/non-finite/boolean diagnostics, feature order/shape/normalization. Phase 1 checks are **non-enforcing**.
5. Evaluate frozen train/validation overall/bucket/semantic/catastrophic metrics; account for missing/review rows.
6. Verify full-core offline/live parity, then application parity with equivalent aggregation/judge/reference state. Residual-only parity is insufficient.
7. Freeze before new final-set access and archive promotion/rejection with evidence.

Source-verified templates: replace placeholders with **new approved data**, not consumed historical sets:

```powershell
python -m scripts.audit_dataset_splits --split-dir <new_materialized_split_directory> --out <new_split_audit_json>
python -m scripts.evaluate_live_pipeline --split-dir <new_materialized_split_directory> --split val --output <new_live_validation_csv>
python -m scripts.evaluate_generalization_manifest --manifest <approved_validation_csv> --images-root . --output-dir <new_readiness_directory>
python evaluate_model_readiness.py --input-dir <approved_image_directory> --manifest <approved_labeled_csv> --manifest-only --enforce-gates
```

Historical readiness defaults: min 50 flood labels/20 depth/25 barren, F1≥0.90, MAE≤15 cm, min 3 labels per depth band, bucket MAE≤8/12/15/20 (0–20/20–50/50–80/80+), barren FP≤0.05, p95 latency≤8 s, contradictions≤0. These are existing script gates, **not newly approved universal promotion thresholds**. Record arguments/read `evaluate_quality_gates()`; smoke sets fail coverage gates. Section 31 adds comparative criteria.

## 16. Legacy Production/Reference Pipeline (V5) Reference Status

V5 freezes Residual Fusion Model (V14) with Phase 1 truthful backend/trace naming, ordered snapshots, per-reference diagnostics, relative-depth naming and non-enforcing contract checks. On **eight approved examples only**, parity to frozen Frozen Predecessor Pipeline (V4) was 8/8, maximum final difference 0.00 cm. Config and checkpoints stayed frozen.

V5 is a controlled-testing/readiness baseline. Audit scope establishes inspectable loading/tracing, not fully solved quality, portable environment, calibrated uncertainty or deployment certification.

## 17. Known Limitations

Current Final Architecture (V6) is strongest roughly at 20–75 cm in the saved 99-row validation analysis. Shallow/no-flood overestimation remains a problem, and 75+ cm needs improvement. These are current validation findings, not final production metrics; see section 36. Meaningful-flood actual >20 cm/prediction <5 cm occurred in 0 V6 versus 5 Legacy Production/Reference Pipeline (V5)/Residual Fusion Model (V14) saved rows, which does not prove that failure is impossible.

The following limitations preserve the previous V5 path findings. Repeated semantic/resolver/guard overwrites apply to V5, not V6 numerical ownership.

- Wet roads/reflections/puddles can resemble flood to water-mask/classifier stages.
- Shallow/ankle 0–20 cm remains difficult: candidate overestimation and residual overcorrection coexist.
- Meaningful flood can collapse to low/zero; identify V14 versus later-guard ownership.
- Deep/extreme flood needs stronger validation; core 180 cm clipping limits represented range.
- Contour references and relative proxy are unreliable direct metric evidence despite retained numerical use.
- Semantics affect numerical depth repeatedly; resolver/high-flood rules can replace good estimates.
- Historical validation has extensive development exposure; consumed final sets are reporting-only.
- Rejected routers are not remedies waiting for a feature flag.
- API reference/judge/aggregation differs from core; static health does not establish readiness.

## 18. Rejected / Experimental Work

[Freeze ledger](reports/architecture_investigation_freeze_20261002.json) records no promotion.

| Experiment | Why inactive / evidence |
|---|---|
| V15 alternate residual | Removing semantic/guard context caused broad regression and failed frozen internal acceptance; `reports/v15_candidate_comparison`, `reports/v15_postmortem` |
| Physical residual/trust two-stage | Controlled train/validation did not support hypothesis; `reports/two_stage_development` |
| Broad 24-image/targeted3 augmentation | No stable multi-seed improvement over clean control; `reports/dev_depth_indian_traffic`, `reports/dev_depth_targeted3` |
| Classifier augmentation/Wet3 | Wet-road benefit not reproducible, wet recall regressed; `reports/dev_classifier_indian_traffic` |
| Collapse-only safety | Validation benefit failed on consumed internal test, dry/wet false positives introduced; `reports/v14_collapse_only_final_eval` |
| Tabular logistic router | Rejected; shadow validation MAE 8.406 versus Residual Fusion Model (V14) 7.601 with bucket tradeoffs; [report](reports/full_pipeline_audit_20261002/phase4_router_experiment/router_experiment_report.md) |
| MobileNetV3 pixel/image router | Rejected; confidently routes known meaningful cases shallow; selective-subset score not full-set gain; [report](reports/full_pipeline_audit_20261002/phase4_image_router_experiment/image_router_report.md) |
| EfficientNet-only shadow | Historical Phase 3 validation tradeoff analysis; its metric ownership is now implemented in Current Final Architecture (V6) shadow architecture. This is not production promotion or completed final training; [report](reports/full_pipeline_audit_20261002/phase3_efficientnet_shadow_report.md) |

Enabled moderate-flood fallback is existing Legacy Production/Reference Pipeline (V5) logic, distinct from rejected collapse-only safety. Older direct-fusion/hierarchical expert reports remain historical evidence. [Next modular architecture](docs/next_modular_architecture_design_20261002.md) is design-only, not implementation.

## 19. Recommended Future Development

All new development uses Current Final Architecture (V6) and the shared entrypoint implementation. Final V6 training on newly collected data remains pending. Keep the present EfficientNet anchor unchanged during UI work. Any future trained refinement requires new data, its own evaluation and acceptance; no existing Residual Fusion Model (V14)/resolver/guard should silently regain numerical ownership.

Collect genuinely new, independently labeled, group-clean data. Prioritize dry/wet/no-flood and 0–20 cm ambiguity while preserving meaningful-depth performance. Label scene and depth separately, add reference-quality evidence, build new development splits and untouched final sets. Avoid endless V14 micro-retraining.

Existing findings favor calibrated semantic context, reliable references, robust physical anchor and bounded refinement. Develop those modules separately on new data. Removing unreliable numerical inputs from Legacy Production/Reference Pipeline (V5) still changes its trained feature distribution: measure deliberately rather than silently changing the reference.

## 20. New Image Collection Guidelines

Collect dry road, wet/no flood, reflections, puddles, 0–10, 10–20, ankle, 20–50, 50–75 and deep flood. Balance similar viewpoints across regimes. Include rain/night/CCTV, Indian traffic, occluded wheels/distant people, broad water with weak/contradictory references.

Preferred labels: scene type, numerical depth cm, confidence, measurement source, source/session ID, physical reference. Record measured image location/time/method/uncertainty; reference wheel/waterline visibility, occlusion, near/mid/far location and reliability. Keep sessions together. Ambiguous or low-confidence depth labels remain review-only, not silently measured regression targets.

## 21. Git / Branch Handover

Keep `flood-model-readiness-guard-v5` frozen as reference/testing branch. Current development uses `flood-model-v6-shadow-dev`; shared-entrypoint commit `7508638` follows Current Final Architecture (V6) UI commit `4c07abb` and cleanup commit `fd60fde`. New experiments should use an isolated branch from the reviewed V6 state. `models/candidate` contains active as well as rejected models.

```powershell
git branch --show-current
git status --short
# Only in a clean checkout after preserving work:
git switch flood-model-v6-shadow-dev
```

Pre-existing local code/data/guide changes mean this workspace differs from a clean committed checkout. None were discarded. Legacy Production/Reference Pipeline (V5) manifest notes wet-road weights are ignored by Git; distribute verified artifact separately unless incorporated through an explicitly reviewed change.

## 22. Troubleshooting

| Issue | Practical checks |
|---|---|
| Missing/wrong checkpoint | Root working directory, exact config path/existence/hash, loader warnings/backend; do not substitute similar basename |
| Unexpected fallback | Reference/dense backends in trace; missing YOLO/HF/dependencies/cache/network/load error changes path |
| Missing packages/load incompatibility | Compare section 25/class/state contract; timm needed for mask fusion, transformers/ultralytics for backends |
| CPU/CUDA differences | Record actual stage devices/versions; configured active stages CPU; replay parity after changes |
| Malformed features | Check checkpoint names/order/dimensions/mean/std, not just Python constant |
| NaN/missing inputs | Inspect `residual_feature_contract_diagnostics`; defaults hide upstream absence; diagnostics not enforcing |
| String boolean | `bool("False")` is true; use actual booleans, explicit CSV conversions/type diagnostics |
| Unexpected zero | Base→raw/applied Residual Fusion Model (V14)→resolver→guards/dry override; inspect status/reasons/corroboration; zero not necessarily dry |
| Offline/live discrepancy | Replay all core stages/backends/hashes, then application policy/state/units |
| Unavailable Current Final Architecture (V6) depth | Shared serialization preserves finite cm or null; zero remains valid. Former SKIPPED_NO_REFERENCE/null metre policy is historical only and is not applied by the current upload API |
| Ready health but missing model | Static `/health`/`/status`; check actual loading logs/trace |
| Config changes ineffective | Restart singleton, inspect resolved overlays/unused fields |
| Quarantined images | Generic loader may move corrupt/checksum/missing-label files; inspect reason/provenance, no automatic recycle |

## 23. Quick Start for a New Developer

If you joined today:

1. Obtain checkout/artifact bundle, confirm branch and preserve local work.
2. Read Current Final Architecture (V6) architecture, UI boundary and parity evidence; read Legacy Production/Reference Pipeline (V5) status/manifest as reference history.
3. Resolve environment discrepancy and record interpreter/packages.
4. Verify the configured EfficientNet primary checkpoint hash, resolved settings and Python packages. The eight-model registry and YOLO/HF assets apply only to historical reference experiments.
5. Read `src/v6_inference.py` and `V6ShadowPipeline.predict()` beside section 4; protect the single EfficientNet owner.
6. Compare an explicitly approved image through V6 entrypoints; the known real-checkpoint smoke is in section 36. Use V5 traces only for reference investigation.
7. Run `python main.py image "<image_path>" --storage local`, Flask `/predict` or `/api/v1/estimate`; each uses the same Current Final Architecture (V6) inference boundary.
8. Read audit/cleanup/parity/reliability/rejection evidence; consumed final data stays reporting-only.
9. Develop new data/modules separately under sections 11/31.

## 24. Key Files Cheat Sheet

| Need to do | File/script to open |
|---|---|
| Current Final Architecture (V6) image→final cm | `src/v6_inference.py`, `src/v6_shadow_pipeline.py:V6ShadowPipeline.predict` |
| Current image/video CLI; optional diagnostics | `main.py`; `scripts/run_v6_shadow_video.py`; optional `scripts/run_v6_shadow_comparison.py` |
| V6 UI presentation | `web_app.py:index()` |
| Legacy Production/Reference Pipeline (V5) core/extractors | `src/segformer_yolo_depthv2_pipeline.py:predict` |
| V6 event adapter and cm output | `src/pipeline.py` |
| Current V6 API upload adapter; stored telemetry queries | `src/api_service.py` |
| CLI/Flask | `main.py`, `web_app.py` |
| Config/overlays | `config/config.yaml`, `src/settings.py` |
| Mask/contour estimates | `src/water_region_detector.py`, `src/reference_depth_estimator.py` |
| One-image trace | `scripts/audit_full_pipeline_trace.py` |
| Frozen readiness/hashes | `reports/flood_model_readiness_guard_v5_manifest.json` |
| Audit/unit findings | `reports/full_pipeline_audit_20261002/full_pipeline_audit_report.md` |
| Consumed sets/rejections | `reports/architecture_investigation_freeze_20261002.json` |
| Image-depth/residual training | `scripts/train_candidate_depth_model.py`, `scripts/train_residual_fusion_depth_model.py` (test-enumeration caution) |
| Development residual | `scripts/run_residual_fusion_depth_dev.py` |
| Data/duplicates | `src/dataset.py`, `scripts/audit_dataset_splits.py` |
| Live evaluation/readiness | `scripts/evaluate_generalization_manifest.py`, `evaluate_model_readiness.py` |
| Proposed architecture | `docs/next_modular_architecture_design_20261002.md` |

## 25. Environment & Dependency Reproducibility

No verified lockfile reconstructs the audited Legacy Production/Reference Pipeline (V5) environment. `.venv/pyvenv.cfg` records CPython **3.14**, uv 0.11.29; installed distribution metadata differs sharply from `requirements.txt`. This is observed local metadata, not proof of exact audited/training interpreter.

| Library | Declared | Observed local distribution metadata |
|---|---|---|
| torch | 2.2.1 | 2.13.0 |
| torchvision | 0.17.1 | 0.28.0 |
| OpenCV | headless 4.9.0.80 | opencv-python 5.0.0.93 |
| transformers | 4.40.0 | 5.14.1 |
| ultralytics | 8.2.0 | 8.4.114 |
| numpy | 1.26.4 | 2.5.1 |
| pandas | ≥2.0.0 | 3.0.5 |
| timm | ≥1.0.0 | 1.0.29 |
| Pillow | ≥10.0.0 | 12.3.0 |

At the earlier V5 documentation audit, metadata was read statically and the local uv launcher failed, so imports were not validated then. On 2026-10-08, real Current Final Architecture (V6) model inference and 23 adapter/regression tests executed successfully using installed CPython 3.14 and the existing environment packages; this does not establish a portable dependency lock. Old pinned wheels on Python 3.14 are not established compatible. Choose/validate an isolated compatible environment, record interpreter and `python -m pip freeze`, then known-image parity. Old README Python 3.8+ is not a validated support range. Training also imports `tqdm`; verify full dependency coverage.

CPU supported/configured; CUDA optional. Stage-specific selectors differ from wrapper auto-detection: Depth Anything device -1, other loaders require config CUDA and availability. GPU parity, CUDA matrix and measured memory requirements are not certified.

Additional declared dependencies include SciPy≥1.10, boto3≥1.26, Flask≥3, Pydantic/Pydantic-settings≥2, PyYAML≥6, Redis≥5 and Celery≥5.3. Their audited installed versions are not established here. Redis connection failure in `SlidingWindowAggregator` falls back to memory: aggregation state becomes process-local, which matters for repeated/multiworker API comparisons.

Verified controls: `FLOOD_APP_ENV`, `FLOOD_CONFIG_PATH`, secret-placeholder variables, `GOOGLE_API_KEY`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`. AWS modes also need external credentials ([S3 setup](S3_SETUP.md), [quickstart](S3_QUICKSTART.md)). Keep secrets out of reports. `SEGMENTATION_BACKEND` belongs to unused alternate engine.

Historical Legacy Production/Reference Pipeline (V5) experiments can contact Hugging Face if caches are absent; that historical depth loader does not pin an immutable revision. Archive resolved cache/model revisions for those experiments. Its failure path activates `dense-depth-proxy`; its former API judge required a network/key. These are historical behaviors and are not invoked by normal Current Final Architecture (V6) inference.

## 26. Input / Output Contract

### Current Final Architecture (V6) contract

`load_v6_rgb(path_or_encoded_bytes)` produces PIL-converted uint8 RGB `(H,W,3)` without entrypoint resizing, normalization or EXIF transpose. Video first saves JPEG and reloads that exact encoded frame. Model transforms remain in the existing model implementation. Browser previews must never replace submitted bytes.

`V6ShadowPipeline.predict()` returns immutable `V6ShadowResult` with `primary_depth_cm`, `final_shadow_depth_cm`, `numerical_owner`, contract, reliability, uncertainty, stages and a historical comparator slot that is null during normal primary-only inference. Current primary and final cm are identical. `v6_depth_payload()` exposes primary cm, final cm and numerical owner, with invalid/nonfinite/missing depths serialized as null. Null means unavailable, not zero. The backend exposes primary depth and numerical owner as well as final depth, but no calibrated confidence score or stage diagnostics. Display formatting may round text only; raw response values remain unchanged.

### Legacy Production/Reference Pipeline (V5) core event and API contracts

**Core input:** one NumPy RGB array `(H,W,3)`, normally uint8 0–255 from PIL `.convert("RGB")`. Shape/channel checks exist, comprehensive dtype/range/size checks do not. OpenCV BGR callers must convert. Wrapper converts grayscale/alpha to RGB. No core batch tensor endpoint; video/audit scripts loop images.

JPG/JPEG/PNG use installed decoders; AVIF is not a verified supported contract, others depend on PIL/OpenCV builds. Config data formats jpg/jpeg/png/gif/bmp and min/max 100×100/4000×4000 are **not enforced core/API request limits**. No explicit EXIF transpose in wrapper/audit path: pre-orient consistently. The historical V5 `/predict` implementation OpenCV-validated bytes then event PIL-decoded them; current `/predict` uses the shared V6 PIL RGB loader directly.

**Core output:** `depth_cm`, `confidence`, `severity`, `method`, `action_trigger`, `water_coverage` fraction, `waterline_pct`, `label_guide`, `visual_cues`, `structured_features`, `pipeline_trace`, `pipeline_stage_outputs`, `depth_teachers`. Guard/review/warning statuses live in features, not universal warnings array. Audit summary uses `final_depth_cm`; core key is `depth_cm`.

Normal finite core outputs are clipped through numerical path to **0–180 cm**, nonnegative. Zero can mean no flood, low-water gate, residual collapse or guard override, not measured zero. Alternate wrapper has different scaling/clamp. Phase 1 does not fully enforce non-finite handling; do not promise arbitrary NaN robustness.

**Event output:** `estimated_depth_meters`, `confidence_score` 0–1, event/trace/camera/time/location, severity/color/action, method/window count, metadata. Aggregation can change image-specific result.

**API:** optional LLM then final decision. Zero reference count produces `SKIPPED_NO_REFERENCE`, `final_decision.estimated_depth_cm: null`, `estimated_depth_meters: null`, preserving raw evidence. Unavailable is not zero. That null-metre failure described the old V5 `/predict`; the current V6 route does not read metre fields. Telemetry can retain raw numeric depth while final decision is null. Inspect status/source/availability; depth fields are not equivalent.

Malformed base64/schema/payload fails validation; PIL decode errors enter retry. `/predict` returns 400 for missing/empty/undecodable upload; estimate catches validation/value errors as 400 and other errors as 500. Wrong core shape raises `ValueError`; no guaranteed successful-depth fallback for corrupt input.

## 27. Model Provenance & Version Registry

All eight hashes below were recomputed locally and match Legacy Production/Reference Pipeline (V5) manifest. Frozen/accepted reference: `flood-model-readiness-guard-v5`, base `874a6b1`; architecture ledger dated 2026-10-02. This is not proof every component passed independent quality acceptance. Exact accepted training dates/commands/datasets are incomplete; never infer from filenames.

| Checkpoint | SHA-256 | Architecture / trainer / dataset-date / status |
|---|---|---|
| `models/best_flood_model_water_aware.pth` | `0418f5dc60bdb4daaf21a9cec1eb00d4e53c64e713e4cf11b4cbc0190453a5fa` | EfficientNet-B0; water-aware trainer family `src/train_water_aware.py`; exact run/split/date unknown. Configured fallback |
| `models/candidate/best_flood_model_water_aware_hardneg.pth` | `78bd165a30b8fee67d17d7ec13b840d6cbdd9d2504445443bf61d52248b5fb40` | EfficientNet-B0 sigmoid/180; compatible candidate trainer; exact hardneg accepted run/split/date incomplete. Active |
| `models/FloodDepth-MaskConditionedFusion.pth` | `e6814a1c6cc704574eefbd4875d94a03c900e73ee3e1618521d9ca5a8fdba862` | EfficientNetV2/object/geometry; full trainer/data/date unavailable; loader verified, stored R²0.297 per audit. Active candidate/correction |
| `models/candidate/no_water_guard_teammate_augmented.pth` | `0dbfb11a4baf2901df4009904fec269c60a2945d4cd0246e2b868b75eedd587d` | Binary MobileNetV3; compatible no-water trainer/class-folder contract, exact run/split/date incomplete. Active |
| `models/candidate/wet_road_no_water_guard_test0916.pth` | `85e4b8afa848fe3f2b53c1dfd93276489bb6cec8332899fe04a0f5618e16efe3` | Binary MobileNetV3, trainer family above; exact run/split/date incomplete, ignored artifact. Active |
| `models/candidate/road_scene_classifier_4class_shallow_v2.pth` | `3763622bfa6c01ff9e7e9330dce8bd28d8ac2c9490230c18805aa23faa8955b9` | Four-class MobileNetV3; compatible scene trainer/fixed splits, exact accepted command/date incomplete. Active |
| `models/candidate/depth_regime_classifier_v6_targeted_conservative.pt` | `f1295ce4a44506c557405fed1586be4d0ff9368245eb0989aa892cdfa6b3b8fd` | EfficientNet-B0+LayerNorm/128 head; fine-tune family, v6 eval/summary reports; exact accepted split/date not reconstructed. Active resolver |
| `models/candidate/residual_fusion_depth_model_ankle_v14_no_leak.pt` | `f4a8dd74aa1b27f04051b33b28ce76bb7dd08cc097b512b3c90b20317a7f0bb8` | `residual_fusion_depth_v1`,40 features,max residual 120,capped-base anchor; residual trainer/v14 feature+eval evidence. Clean 465/99 audit split not automatic original-training provenance. Active Residual Fusion Model (V14)/V5 |

Config SHA-256: `6a5de50beb813fea30eba23cb460f455199276954d8c91d84ffec0cfca2ba7dd`. External YOLO/HF immutable hashes/revision are absent from eight-weight ledger; future registry must add them. New records need trainer revision/command, train/val hashes, label provenance, seed, environment, preprocessing/features, report, acceptance branch/date and hash.

## 28. Reproducibility Checklist

- [ ] Correct branch/revision and recorded dirty state
- [ ] Correct raw/resolved configuration
- [ ] All checkpoints/assets present
- [ ] Matching model/config hashes
- [ ] Required Python/packages validated and recorded
- [ ] Correct segmentation/reference/dense/residual backend
- [ ] No unexpected fallback/unavailable component
- [ ] HF cache revision recorded
- [ ] Known-image smoke checked against saved baseline
- [ ] Trace generated and contract diagnostics reviewed
- [ ] Application/aggregation/judge distinguished from core parity

Read-only PowerShell check:

```powershell
$registry = Get-Content reports/flood_model_readiness_guard_v5_manifest.json -Raw | ConvertFrom-Json
$registry.configured_active_checkpoints.psobject.Properties | ForEach-Object {
    $actualHash = (Get-FileHash -LiteralPath $_.Name -Algorithm SHA256).Hash.ToLower()
    [pscustomobject]@{ Path = $_.Name; Matches = ($actualHash -eq $_.Value) }
}
Get-FileHash config/config.yaml -Algorithm SHA256
```

## 29. Deployment / Runtime Runbook

**Deployment procedure not yet formally defined.** YAML AWS/ECS/Lambda/LitServe settings are not a verified production deployment/restart service unit or capacity plan.

Current Final Architecture (V6) UI start: `python web_app.py`, then `http://localhost:5000`. The equivalent launch is `python main.py web --host 127.0.0.1 --port 5000`; `main.py` image/video modes now use Current Final Architecture (V6). `python web_app.py` starts Flask 0.0.0.0:5000/debug, a development server rather than deployment recipe. Initialization is lazy at the first prediction and loads the configured local EfficientNet checkpoint. Measured startup time/RAM/VRAM/safe worker count are not formally documented; YAML 2 vCPU/4GB is not measured capacity.

Normal Current Final Architecture (V6) prediction needs the configured EfficientNet checkpoint, resolved configuration and Python packages. Local image inference does not require the historical eight-weight bundle, YOLO/HF caches, a judge key or Redis. S3 input requires configured AWS access. Flask stored-telemetry routes retain their SQLite dependency; new V6 uploads do not write legacy telemetry. CPU is configured; GPU is optional. Logging uses the console/Python logger and event trace IDs; `monitoring.log_dir` does not establish universal file logging.

Local restart: Ctrl+C, verify the configured checkpoint/config, then rerun the command. Production restart is undefined. `/health` and `/status` report architecture V6 and remain static availability responses; they do not validate model initialization. Verify readiness with an approved-image prediction and checkpoint/RGB hashes. Prediction failures include invalid input, missing dependencies or checkpoint/state mismatch; S3 and stored-telemetry routes have their own AWS/SQLite dependencies.

## 30. Rollback Procedure

1. Preserve experimental/local work and deployment artifacts; use clean checkout/worktree.
2. Select an explicitly reviewed Current Final Architecture (V6) revision that retains the shared inference boundary; do not switch normal inference to the historical V5 branch.
3. Restore that revision’s recorded V6 configuration/checkpoint and verify its hashes/environment.
4. Restart and run approved entrypoint/RGB parity checks with equivalent inputs.
5. Redeploy using mechanism once defined; verify service health and actual readiness; record revision/hashes/logs.

Historical reference restoration may use `flood-model-readiness-guard-v5` and its recorded artifact bundle only for an explicitly scoped historical experiment. There is no automatic V5 fallback in normal inference. Experiments must never overwrite historical artifacts. Git switch does not restore ignored weights/cache/packages/DB/window state or reload singleton models.

## 31. Acceptance Criteria for a Future Model

No new numerical pass/fail thresholds are invented. Historical script gates differ from a formally approved comparative promotion policy. Review:

- No duplicate/session/label leakage, group-clean frozen validation, independent label provenance.
- No tuning/selection/architecture decisions using consumed internal/challenge sets.
- Same-row overall MAE/median AE/RMSE/within±5/10 cm/worst and regression lists.
- Depth buckets 0–10/10–20/20–30/30–50/50–75/75+, scene confusion/calibration, shallow/deep coverage.
- Severe under/overestimation with evaluator definition recorded.
- Actual >20→prediction <5 and actual >30→prediction <10 counts.
- Review/unavailable/NaN rows retained in accounting; no selected-subset gain claims.
- Full-core offline/live parity, separately application/units/aggregation/judge parity.
- Config/model/source/environment/data hashes and actual active backends.
- New untouched final sets used once after freeze; rejection does not permit tuning on results.

One repaired image/lower residual-only MAE/router subset is insufficient. Record tradeoffs, data limits and explicit acceptance before replacing paths.

## 32. Failure Taxonomy

Record first failing stage plus secondary contributors:

| Category | Meaning |
|---|---|
| SEGMENTATION FAILURE | Wrong water mask/zones/reflections/missed muddy water |
| SEMANTIC FAILURE | Wrong no-water/scene/regime probabilities |
| DEPTH MODEL FAILURE | Wrong supervised candidate with valid input |
| REFERENCE FAILURE | Bad detection/submersion/contour cm or count-depth inconsistency |
| FUSION FAILURE | Plausible candidates become incorrect base/agreement |
| RESIDUAL FAILURE | Raw/applied Residual Fusion Model (V14) damages better input, including collapse |
| RESOLVER FAILURE | Later resolver incorrectly replaces/raises output |
| GUARD FAILURE | Cap/zero/override suppresses flood or permits false flood |
| BACKEND/FALLBACK FAILURE | Missing/load-failed component or unexpected proxy/environment |
| LABEL/DATA ISSUE | Label ambiguity/error/orientation/color/duplicates/leakage/provenance |

Include stage depths/status/reasons/backend/hash/label confidence. Record application reference/judge/aggregation owner separately rather than attributing all wrong outputs to V14.

## 33. Data Labeling Standard

No universal extended schema is enforced today. Adopt this explicit **proposed collection convention**, then map to trainer schemas. Current loaders do not validate all these fields:

```text
image_id: stable unique ID
filename: exact filename
image_path: artifact-relative path
scene_type: DRY | WET_NO_FLOOD | SHALLOW_FLOOD | MEANINGFUL_FLOOD
depth_cm: nonnegative numeric cm; blank for unknown/untrusted
label_confidence: HIGH | MEDIUM | LOW
label_source: MEASURED | DOCUMENTED | VISUAL_ESTIMATE
source_session_id: stable video/burst/location-session/source ID
physical_reference: PERSON | CAR | BIKE | CURB | OTHER | NONE
sha256: exact image hash
measurement_notes: location/method/time/uncertainty/evidence
```

Scene and numerical labels are separate. Scene trainer uses lowercase `dry_road,wet_road,shallow_flood,meaningful_flood`; explicitly map enums. Phase 4 binary analysis uses≤20 versus >20 cm, but no universal enforced collection-wide ankle/shallow policy exists: freeze policy before labeling. Mixed/ambiguous scenes remain review-only.

**Filename cm is not automatically ground truth.** Independently verify encoded depth; keep original filename as metadata. Hybrid/filename/auto-label utilities infer labels, not measured targets. Unknown is not zero. Preserve confidence/visual-estimate provenance; ambiguous labels must not silently enter regression.

## 34. What NOT to Do

- Randomly retrain Residual Fusion Model (V14) after each bad image.
- Tune/select thresholds/models/architecture on consumed internal/challenge sets.
- Mix duplicates/sessions across splits or treat uncertain singletons as proof of independence.
- Assume filename cm is true or unknown depth is zero.
- Treat normalized Depth Anything/unreliable contour cm as trusted metric evidence.
- Attribute separate contour cm to YOLO reference count.
- Enable rejected routers/archived safety candidates as production fixes.
- Replace weights without hashes/provenance/contract/acceptance evidence.
- Repeatedly tune validation thresholds and call it independent evidence.
- Interpret one good image/selected subset as model improvement.
- Confuse raw/applied V14/final core/aggregate event/API final decision.
- Treat static health/matching branch/local hashes alone as complete parity.

### Documentation inventory and unresolved gaps

Identified before writing: `README.md` (competing legacy end-to-end overview, replaced with pointer), `SETUP.md`, `S3_SETUP.md`, `S3_QUICKSTART.md`, `models/README.md`, `docs/MODEL_TRAINING_GUIDE.md`, `RESIDUAL_FUSION_TRAINING.md`, `docs/PRODUCTION_READINESS_STATUS.md`, `docs/next_modular_architecture_design_20261002.md`. No dedicated master KT existed at the original authoring date; this update edits the existing Markdown/Word MASTER KT rather than creating another handover. Supporting guides and audit/experiment/readiness evidence were retained.

Incomplete evidence: exact accepted training run/split/date for several weights; full mask-fusion trainer/provenance; audited interpreter/dependency lock; external-model immutable registry; production deployment/restart/capacity; universally enforced label schema/promotion thresholds; application parity scope. The earlier local Python launcher failure is historical; Current Final Architecture (V6) real-checkpoint execution-path parity is documented in section 36. Fill gaps with evidence without altering frozen reference.

## 35. Current Final Architecture (V6) UI Architecture and Team Boundary

The existing UI is **web_app.py**, not webapp.py. It uses **Flask with inline HTML/CSS/JavaScript** in `index()`; the title is **Flood Depth Estimator – V6**. It currently has one image picker, Analyze Image button, inline loading/error text and a cm result. No preview, video UI, history table, spinner or numerical progress is implemented.

The exact UI call chain is:

```text
index()
 → browser POST /predict with original multipart image
 → request.files["image"].read()
 → load_v6_rgb(image_bytes)
 → get_v6_pipeline()
     → create_v6_pipeline() on first use
 → V6ShadowPipeline.predict(image_rgb)
 → v6_depth_payload(result)
 → jsonify(v6_depth_payload(result))
 → browser displays final_shadow_depth_cm in cm
```

`/api/v1/estimate` now uses the shared V6 event adapter. Camera stats, stored telemetry and temporal analysis remain historical-data services and never own or replace V6 predictions. The UI continues using `/predict` and never uses `estimated_depth_meters`. Health/status report V6 but remain static availability responses, not proof that a checkpoint loaded.

### Safe presentation work

Safe area: **web_app.py → index()**. HTML, CSS and presentation JavaScript may change layout, tabs/sidebar, buttons, colors, headings, cards, tooltips, drag/drop, preview, loading spinner, reset, browser history, downloadable returned results and responsiveness. Keep raw cm values for history/downloads; round only display text. Use a spinner rather than invented stage percentages: the backend returns one final response.

Required contract: **POST /predict**, **multipart field = image**, **output field = final_shadow_depth_cm**. Null means unavailable and zero is valid. Preview/resizing/cropping/compression must never replace or alter the original inference bytes. Interactive controls must not change model parameters, add thresholds, corrections, fusion, guards, resolver or averaging, or construct another prediction path.

### Protected inference work

Do not change `web_app.py → predict()` or `get_v6_pipeline()`, `src/v6_inference.py`, `src/v6_shadow_pipeline.py`, `src/v6_shadow_contract.py`, shared comparison/serialization, or model/config/checkpoint loading during UI work. The submit listener is shared/careful: change feedback while preserving endpoint/method/field/original bytes. Do not weaken parity assertions.

```text
UI HTML CSS preview controls                         SAFE UI
 → browser upload event                              SHARED / CAREFUL
 → Flask predict and shared V6 input boundary        DO NOT MODIFY
 → V6ShadowPipeline.predict                          DO NOT MODIFY
 → final_shadow_depth_cm and shared serializer       DO NOT MODIFY
 → UI display cards formatting history               SAFE UI
```

Reliability, uncertainty and stages exist internally. `/predict` exposes primary cm, final cm and numerical owner through the shared serializer, without a calibrated confidence score. Adding diagnostic response fields requires separately reviewed backend/serialization work, not UI-only edits.

| File or function | Purpose | Classification |
|---|---|---|
| web_app.py index HTML and CSS | Layout, labels, cards and responsive styling | SAFE UI |
| Browser result/preview/history/reset/download code | Present original image and returned results | SAFE UI |
| Browser submit listener and create_app wiring | Preserve upload and application lifecycle | SHARED / CAREFUL |
| web_app.py predict and get_v6_pipeline | Decode, call/cache predictor, serialize | DO NOT MODIFY |
| src/v6_inference.py | Factory, RGB and finite/null output contract | DO NOT MODIFY |
| V6 pipeline/contract/comparison modules | Numerical ownership, fields and serialization | DO NOT MODIFY |
| Models/config/settings and extractors | Checkpoint identity, preprocessing and behavior | DO NOT MODIFY |
| V6 runners and upload API routes | Parity-tested shared orchestration | DO NOT MODIFY |
| Parity tests | Protect shared execution contract | SHARED / CAREFUL |

Source guide: [V6 entrypoint architecture](docs/V6_ENTRYPOINT_ARCHITECTURE.md).

## 36. Current Final Architecture (V6) Validation and Entrypoint Parity

### Current validation findings

These are **current validation results, not final production metrics**. The saved Phase 3 analysis replays the EfficientNet candidate from an existing 99-row validation feature table; no model was rerun for that report. Current V6 has exactly this numerical ownership. This is not a fresh independent V6 evaluation or accuracy evidence for the pending new dataset.

| Metric | V6 EfficientNet anchor |
|---|---:|
| Validation size | 99 |
| MAE | 9.17 cm |
| Median absolute error | 5.12 cm |
| RMSE | 13.04 cm |
| Within ±5 cm | 48.48% |
| Within ±10 cm | 66.67% |

| Actual depth bucket | Rows | V6 MAE |
|---|---:|---:|
| 0–10 cm | 14 | 18.41 cm |
| 10–20 cm | 16 | 11.66 cm |
| 20–30 cm | 31 | 4.91 cm |
| 30–50 cm | 20 | 6.77 cm |
| 50–75 cm | 8 | 6.88 cm |
| 75+ cm | 10 | 12.11 cm |

V6 is strongest roughly at 20–75 cm. Shallow/no-flood overestimation remains a problem; very deep 75+ cm performance needs improvement. In the saved same-row comparison, **actual >20 cm → prediction <5 cm: Legacy Production/Reference Pipeline (V5)/Residual Fusion Model (V14) = 5 cases; V6 = 0 cases**. This is a sample result, not a guarantee.

Verified sources: [Phase 3 summary](reports/full_pipeline_audit_20261002/phase3_efficientnet_only_summary.json) and [shadow analysis](reports/full_pipeline_audit_20261002/phase3_efficientnet_shadow_report.md). Exact saved MAE/RMSE are 9.173/13.039 cm, rounded above.

### Legacy Production/Reference Pipeline (V5) benchmark caveat

Keep historical V5/V14 MAE **7.60 cm** as a development result, not clean production accuracy. The provenance finding supplied for this handover is that **77/99 validation images had appeared in the original V14 training population**. Accordingly, 7.60 cm is not an independent benchmark. The reported clean-retraining baseline was approximately **9.3–9.4 cm MAE**.

The 7.601 cm score is verified in the Phase 3 summary. The exact 77/99 original-population overlap and specific clean-retraining run behind 9.3–9.4 cm require the corresponding provenance/run artifact before independently reproducing those figures. A group-clean later split does not retroactively remove a checkpoint's original training exposure. Never present 7.60 cm as clean production accuracy.

### Real trained-checkpoint execution parity

Commit **7508638** established shared V6 construction/RGB/output parity. The default migration was rechecked using the same approved inputs and real checkpoint. The approved smoke used `066_20CM.jpg` and the first saved JPEG frame from the previously tested DAV. It ran the actual image CLI, Flask upload route, V6 video saved-frame function, and independent image CLI on that same JPEG with real configured models.

Configured primary checkpoint: **models/candidate/best_flood_model_water_aware_hardneg.pth**. Verified SHA-256: `78bd165a30b8fee67d17d7ec13b840d6cbdd9d2504445443bf61d52248b5fb40`. Every path verified loaded eval-mode EfficientNet. Instrumentation recorded real prediction results without replacing inference.

| Execution path | primary_depth_cm | final_shadow_depth_cm |
|---|---:|---:|
| Default main.py image CLI | 7.68 | 7.68 |
| Shared direct / diagnostic image CLI | 7.68 | 7.68 |
| UI/backend same image | 7.68 | 7.68 |
| Video saved frame | 77.39 | 77.39 |
| Independent CLI same saved JPEG | 77.39 | 77.39 |

**Exact differences are 0.0 cm for both fields in each pair**, before display formatting. Decoded RGB hashes matched within each pair. This proves **execution-path parity, not model accuracy**. It does not certify all formats, environments, GPU execution or deployment capacity. No accuracy evaluation or retraining was performed in the smoke or this KT update.

Local evidence: [parity results](reports/v6_entrypoint_parity_smoke/parity_results.json), `reports/v6_entrypoint_parity_smoke/image_cli.json`, `reports/v6_entrypoint_parity_smoke/saved_frame_cli.json`, and `reports/v6_entrypoint_parity_smoke/video/frames/frame_000000.jpg`. These local artifacts are not part of commit 7508638; the committed [entrypoint guide](docs/V6_ENTRYPOINT_ARCHITECTURE.md) records the results. Earlier adapter validation reported 23 tests. The default migration passed 46 regression tests plus one real-checkpoint parity test. Current local evidence: `reports/v6_default_entrypoint_parity_smoke/parity_results.json`.

## 37. V6 Video Input Pipeline – DAV Support and Deployment Dependencies

```text
video
 → OpenCV decode
 → FFmpeg fallback if OpenCV yields no valid frame
 → selected frame
 → save JPEG
 → reload saved JPEG as RGB through load_v6_rgb
 → same V6ShadowPipeline.predict image implementation
 → per-frame predictions.csv
 → run_summary.json
```

`src/v6_video_input.py` owns decoding only. `scripts/run_v6_shadow_video.py → process_saved_frames()` saves each selected BGR frame, reloads the JPEG with shared PIL RGB, then calls the predictor. `load_saved_rgb` is a compatibility alias of `load_v6_rgb`, not a second preprocessor. JPEG encoding is part of this video contract: compare image inference with the exact saved JPEG, not the unencoded frame array.

The tested DAV was **Dahua dhav**, **HEVC/H.265**, **1280×1440**, average **15 FPS** (nominal 25 FPS). FFmpeg decoded the first **100 frames**; OpenCV also decoded this DAV. FFmpeg remains fallback. Duration was unavailable in inspected metadata. This validates that file, not every CCTV/DVR variant.

FFprobe inspects stream dimensions/FPS and metadata; FFmpeg decodes raw BGR frames. **Both executables must be on PATH in EC2/Docker when DAV or other fallback decoding is required**, with the required demuxer/codec support. OpenCV-decodable inputs do not require fallback binaries for that path. No EC2/Docker deployment is certified by local tests.

Source-verified commands:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_v6_shadow_video --video "C:\videos\camera.mp4" --output-dir "reports\v6_video_mp4" --skip-frames 1
.\.venv\Scripts\python.exe -m scripts.run_v6_shadow_video --video "C:\videos\camera.dav" --output-dir "reports\v6_video_dav" --max-frames 2 --skip-frames 1
ffmpeg -version
ffprobe -version
```

For future Ubuntu EC2/Docker, provision FFmpeg/ffprobe in the server/image build and verify binaries and actual decoding before video ingestion. Do not rely on a developer's Windows PATH. Encrypted, damaged or vendor-specific files can still fail.

V6 records open/decode failures, unavailable binaries, failed probing, bad/truncated frames and valid frames with unavailable depth. Bad saved-frame/inference errors are recorded per frame; unavailable depth is not used in arithmetic. Tests cover these paths and none/nonfinite handling.

Evidence: [DAV validation](reports/v6_shadow_architecture_readiness.md), `reports/v6_dav_smoke_100frames.csv`, `reports/v6_dav_smoke_2frames.csv`, and `reports/v6_video_runs/dav_smoke_2frames/run_summary.json`.

## 38. Current Final Architecture (V6) Training Data Readiness and Repository Cleanup

### Training readiness

**Final V6 training on the new dataset has not yet been completed.** Current V6 reuses the existing configured EfficientNet checkpoint. The intended process is:

```text
new images → manifest → label validation → duplicate/provenance audit
 → group-safe split → freeze challenge → V6 training → final evaluation
```

`templates/v6_new_image_manifest.csv`, `src/v6_training_data_contract.py` and `scripts/prepare_v6_training_dataset.py` provide label validation, hash/session grouping, optional visual duplicate checks, quarantine, deterministic group-safe splits and challenge freeze. Visual checks without images are marked not_run, not passed. Dry/wet-no-flood rows are classification-only with blank depth; missing depth never becomes zero.

`configs/v6_training_skeleton.yaml` is a placeholder, not a completed runnable training recipe. `scripts/evaluate_v6_future_template.py` consumes precomputed predictions; final Challenge mode requires a frozen candidate artifact. Synthetic tests establish infrastructure behavior, not real-data/model acceptance.

**Previously consumed internal-test and external-challenge sets are reporting-only. Do not reuse them for tuning, model selection, retraining decisions or architecture design.** Freeze genuinely new final sets before training and access them only after candidate freeze. Preserve Legacy Production/Reference Pipeline (V5) as reference while all new development uses V6.

Evidence: [V6 training readiness](reports/v6_training_readiness.md).

### Repository cleanup

Cleanup commit **fd60fde** changed only `.gitignore`, `docs/REPOSITORY_CLEANUP_AUDIT_20261007.md` and `docs/REPOSITORY_CHECKPOINT_USAGE_20261007.json`. Old timestamped reports were archived locally under ignored `archive/reports/legacy_timestamped/`; they were not added to the commit. Active checkpoints, source, config, data and KT were preserved. Cleanup did not change predictions.

| Smoke path | Before cm | After cm |
|---|---:|---:|
| V5 image | 0.00 | 0.00 |
| V6 image | 22.95 | 22.95 |
| V6 video | 25.85 | 25.85 |

The video smoke was synthesized and JPEG-encoded from the approved image; its different depth reflects different encoded input, not cleanup. Before/after comparison/backend payloads were equal. This is engineering parity, not accuracy improvement.

Evidence: [cleanup audit](docs/REPOSITORY_CLEANUP_AUDIT_20261007.md), [checkpoint usage](docs/REPOSITORY_CHECKPOINT_USAGE_20261007.json), local `__agent__/repo_cleanup_20261007/before_smoke.json` and `__agent__/repo_cleanup_20261007/after_smoke.json`. Ignored checkpoints/datasets/caches/videos/QA artifacts require a separate approved distribution bundle. Unrelated dirty-tree changes and image deletions must not be staged with documentation.

## 39. KT Update Audit and Evidence Limits

This 2026-10-08 update documents the V6-default migration. Frozen Predecessor Pipeline (V4)/Legacy Production/Reference Pipeline (V5) stage/training/checkpoint/rejection material is preserved with explicit reference scope. Current Final Architecture (V6) overview, shared architecture, UI boundary, commands, output contract, validation/parity, DAV support, training readiness and cleanup are updated. Normal CLI/API/event execution now uses shared V6 only. EfficientNet loading/transform/scaling/rounding were mechanically extracted without changing model behavior; checkpoints, configuration and training are unchanged. Only migration files, tests and documentation are staged; unrelated workspace changes remain excluded.

Corrected statements include V5 as current development branch, `/predict` calling FloodApiService, UI metre conversion/null arithmetic, EfficientNet only being a proposed anchor, V6 video using --output, and the earlier launcher failure implying no current inference verification.

Evidence limits: exact 77/99 original Residual Fusion Model (V14) overlap and approximately 9.3–9.4 cm clean-retraining run are supplied provenance findings requiring their source artifact; no fresh accuracy evaluation was performed. Accepted training provenance for several historical weights, immutable external revisions, portable environment lock, GPU parity, production capacity/restart procedure and final new-data training/evaluation remain incomplete. Local/ignored artifacts may exist here without being available in a clone.

Path audit checked 134 concrete referenced paths. The historical training_data/un_labled_images folder is absent and is explicitly labeled historical in section 9; all other checked paths exist locally. Referenced paths are checked against this workspace. Placeholder arguments, remote Hugging Face IDs and prospective deployment paths are not existing-file claims. Being active in V5 does not give a checkpoint V6 numerical authority.
