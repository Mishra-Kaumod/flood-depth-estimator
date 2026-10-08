# Current Final Architecture (V6) entrypoint architecture

Audited and validated on 2026-10-08 on flood-model-v6-shadow-dev.

**Current Final Architecture (V6) is the only active prediction architecture. Legacy Production/Reference Pipeline (V5) is retained only for historical/reference purposes and is not part of normal inference.**

## Recommended execution

```sh
python main.py image "<image_path>" --storage local
python main.py video "<video_path>" --storage local --output video_analysis.csv --skip-frames 1
python main.py web --host 127.0.0.1 --port 5000
```

Image output prints Flood Depth Estimator – V6 and Estimated Flood Depth: XX.XX cm, followed by primary/final cm and numerical-owner JSON. Camera/location arguments remain accepted for compatibility. Storage defaults to AWS; local/S3 changes retrieval only. The image command no longer runs event ingestion or writes the legacy final-prediction CSV/Parquet.

## Shared boundary

```text
main.py image / Flask UI / diagnostic image CLI / video CLI / API / worker / EC2
 → src.v6_inference.load_v6_rgb
 → src.v6_inference.create_v6_pipeline
 → V6ShadowPipeline(EfficientNetDepthSignal()).predict(image_rgb)
 → primary_depth_cm = final_shadow_depth_cm
 → src.v6_inference.v6_depth_payload
```

The shared contract is PIL-decoded uint8 H×W×3 RGB. Adapter code never resizes, normalizes, corrects, fuses, thresholds or averages prediction. Video saves a JPEG then reloads it through that loader; compare with the exact saved JPEG, not the pre-encoding array. None/NaN/infinite depth becomes null/unavailable, never zero. Invalid images and checkpoint failures produce explicit errors; no V5/random-weight fallback exists.

The EfficientNet builder, checkpoint loading, eval mode, device choice, transform, max-depth scaling and rounding were mechanically extracted unchanged from the legacy core into src/efficientnet_depth_signal.py. The legacy core inherits the same methods. The active factory loads only this primary model and never constructs the full legacy pipeline. Optional legacy context diagnostics are missing/reporting-only in normal V6 results.

| Entrypoint | Input orchestration | Prediction/output |
| --- | --- | --- |
| main.py image | Local/S3 encoded bytes, shared RGB | Shared factory/predict/payload; cm console and JSON |
| web_app.py /predict | Original uploaded bytes, shared RGB; lazy cached pipeline | Shared factory/predict/payload; HTTP 200 with primary/final/owner |
| /api/v1/estimate | Existing multipart/base64 camera contract | FloodApiService → execute_event → shared V6; synchronous HTTP 200 result |
| src/worker.py process_camera_event | Existing base64/event input | Same V6 event adapter; no legacy aggregation |
| src/pipeline.py execute_event | Event metadata, retry/observability | Shared V6 only; no alternative predictor |
| scripts/run_v6_shadow_comparison.py | Approved image/hash verification | Optional diagnostic stages; V5 comparator fields null |
| main.py video / scripts/run_v6_shadow_video.py | Existing V6 decoder/save-JPEG/reload-RGB flow | Shared V6 per saved frame; primary/final cm CSV |
| EC2 / future API | Same commands or event adapter | Must use this shared boundary; no distinct model implementation |

## Audit classification and compatibility

Former main image/video, API upload and queue/event paths were active legacy callers and are migrated. The default V6 factory's hidden full-V5 construction was also removed. Historical/reference callers retained: scripts/audit_full_pipeline_trace.py, scripts/evaluate_generalization_manifest.py, scripts/evaluate_live_pipeline.py, scripts/evaluate_frozen_collapse_safety.py, scripts/evaluate_residual_safety_fallback.py, scripts/compare_mask_conditioned_fusion.py, evaluate_model_readiness.py, scripts/auto_fill_labels.py (old labeling experiment), scripts/train_fusion_depth_model.py (old feature extraction/training), and archived CLI modules. They are not recommended/default inference paths. No new legacy selector was added. Object detection is a separate helper, not a flood-depth predictor.

Camera stats, stored telemetry and temporal analysis are unrelated historical-data services, retained without changing their algorithms. Temporal analytics can summarize old stored records; those outputs never feed V6 prediction. New V6 uploads do not apply reference requirements, judge corrections, severity or automatic aggregation. They do not write the old SQLite telemetry schema, whose required confidence/severity values V6 does not provide. Clients must use cm fields and HTTP 200 rather than former metre/final_decision/queue fields and HTTP 202. Health/status are availability responses, not checkpoint readiness assertions.

The old main flow was main.process_image_cli → FloodApiService.process_camera_upload → execute_event → legacy fusion → aggregation/judge/reference gate. The new image flow is encoded bytes → shared RGB → shared V6 factory → predict → shared cm payload. No active/default V5 prediction caller remains.

## Validation

46 regression tests cover V6 contracts, ownership, video decoding/saved frames, UI/CLI parity, main image/video, event/API/worker parity, invalid inputs, unavailable/nonfinite output, model failure and exclusion of the legacy constructor. One opt-in real-checkpoint smoke also passed. Run the bounded real smoke explicitly with FLOOD_RUN_REAL_V6_PARITY=1; it uses previously approved local inputs and evidence and skips in ordinary CI.

Real primary checkpoint: models/candidate/best_flood_model_water_aware_hardneg.pth; SHA-256 78bd165a30b8fee67d17d7ec13b840d6cbdd9d2504445443bf61d52248b5fb40. Approved 066_20CM.jpg SHA-256 f35019bbd4bc8d9721a035d9a7d817937a1113ef64940a0acda1269db7eb30fa.

| Path | primary_depth_cm | final_shadow_depth_cm |
| --- | ---: | ---: |
| main.py image | 7.68 | 7.68 |
| Shared direct / diagnostic image CLI | 7.68 | 7.68 |
| UI/backend | 7.68 | 7.68 |
| DAV saved frame | 77.39 | 77.39 |
| Independent main.py image on that JPEG | 77.39 | 77.39 |

Every difference is exactly 0.0 cm before display rounding. PIL RGB hashes match within image/video pairs and prior commit 7508638 evidence; the saved JPEG byte hash also matches. Image RGB SHA-256 b2d6302c8f8d8c413fc8f429fe8ac344382e02aed8eee2e960d40d3cc5a14062; saved-frame RGB SHA-256 87ad2124a054748be39593b81a9c4e068164ff8d055bf0456f90752c98d91030. Local evidence is under reports/v6_default_entrypoint_parity_smoke and is not staged. This verifies execution parity only; no accuracy evaluation, training or checkpoint/config change was performed.
