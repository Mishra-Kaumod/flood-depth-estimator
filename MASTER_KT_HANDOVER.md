# Flood Depth Estimator Project Handover

**Snapshot:** 10 October 2026 | **Branch:** `flood-model-v6-shadow-dev` | **Latest output-cleanup commit:** `2f35728`

This handover describes the current V6 implementation and the steps needed to run and maintain it. Source code and `config/config.yaml` remain authoritative when local settings differ. A completed inference is an estimate; it is not an accuracy certification or a surveyed measurement.

**Current Final Architecture (V6) is the only active prediction architecture. The Legacy Production/Reference Pipeline (V5) and its Residual Fusion Model (V14) are historical/reference material only.**

## Contents

The following restores the earlier 33-topic handover index. The current guide keeps detailed active implementation guidance consolidated in its numbered sections; listing a topic here does not imply that historical or training material is part of active V6 inference.

1. Project Overview
2. Repository Structure
3. Starting Point Application Entry Point
4. End to End Runtime Pipeline
5. Python File by File Explanation
6. Configuration
7. Active Models and Checkpoints
8. Important Signal Contracts
9. Data and Dataset Structure
10. Training Architecture
11. How to Retrain Fine Tune in Future
12. Training Scripts Reference
13. How to Run Prediction
14. How to Debug One Wrong Prediction
15. Model Readiness Process
16. Current V6 Status
17. Known Limitations
18. Rejected Experimental Work
19. Recommended Future Development
20. New Image Collection Guidelines
21. Git Branch Handover
22. Troubleshooting
23. Quick Start for a New Developer
24. Key Files Cheat Sheet
25. Environment Dependency Reproducibility
26. Input Output Contract
27. Model Provenance Version Registry
28. Reproducibility Checklist
29. Deployment Runtime Runbook
30. Rollback Procedure
31. Acceptance Criteria for a Future Model
32. Failure Taxonomy
33. Data Labeling Standard

## 1. Terminology and current status

| Descriptive name | Internal version | Meaning |
| --- | --- | --- |
| Current Final Architecture | V6 | Active metric-depth architecture with EfficientNet as the immutable primary anchor. |
| Shared V6 Inference Layer | V6 | Common RGB loader, factory, pipeline call, and result contract used by image, UI, API, event, and video entrypoints. |
| Frozen Predecessor Pipeline | V4 | Earlier implementation retained as historical material. |
| Legacy Production/Reference Pipeline | V5 | Previous multi-stage implementation retained for reference; not used for normal inference. |
| Residual Fusion Model | V14 | Historical residual model associated with the legacy pipeline; it does not own V6 output. |

V6 gathers available water, scene, object, semantic, and relative-depth context. These secondary signals are diagnostic or experimental. The single internal correction authority currently abstains because no secondary metric candidate has an approved acceptance rule. Therefore the controlled V6 final remains the EfficientNet primary result. Optional Gemini review runs after V6 and may change only the application-level result when explicitly enabled and a valid correction is returned.

## 2. Quick start

Run commands from the repository root in PowerShell. Activate the project environment if it is not already active.

**Image CLI**

```powershell
.\.venv\Scripts\Activate.ps1
python main.py image "C:\path\to\image.jpg" --storage local
```

Add `--verbose` for a short diagnostic summary or `--debug` for the complete result JSON in the console. Debug output is also written to the JSON sidecar described in section 5.

**Web application**

```powershell
python main.py --app
```

Open `http://127.0.0.1:5000`. The browser upload form is `/predict`. Health is available at `/health`.

**Video CLI**

```powershell
python main.py video "C:\path\to\clip.mp4" --storage local --output reports\video_predictions.csv
```

Use `--skip-frames N` to override the configured frame interval. The V6 video input flow saves sampled frames as JPEGs, reloads those JPEGs as RGB, then calls the same shared V6 pipeline. Gemini review for video frames is disabled by default.

## 3. Runtime architecture and result ownership

All active prediction adapters converge on the same inference implementation. They may differ in how they receive bytes, metadata, or saved frames; they do not own separate depth-estimation logic.

```text
Image CLI / UI upload / API / worker-event / EC2 event / video frame
                         ↓
        shared V6 RGB input and pipeline factory
                         ↓
              V6ShadowPipeline.predict()
                         ↓
   EfficientNet primary + typed diagnostic evidence
                         ↓
     one controlled correction authority (abstain)
                         ↓
              final_shadow_depth_cm
                         ↓
         optional application-level Gemini review
                         ↓
              application_final_depth_cm
```

The image path is `main.py → load_v6_rgb() → create_v6_pipeline() → V6ShadowPipeline.predict() → review_v6_result() when the reviewer module is available → report/export`. The UI, API/event adapter, and video runner use the shared V6 factory and prediction boundary as well. EC2 execution uses the same event/worker adapter; there is no separate EC2 depth model.

| Result field | Owner and meaning |
| --- | --- |
| `primary_depth_cm` | Original EfficientNet metric result; immutable. |
| `final_shadow_depth_cm` | Controlled V6 final in centimetres. It is preserved even if an application review runs. |
| `final_v6_depth_cm` | Serialized alias for the controlled V6 final. |
| `numerical_owner` | `efficientnet_primary_anchor` for the V6 metric result. |
| `application_final_depth_cm` | Final value returned to the application. Equals the V6 final unless an allowed Gemini correction is applied. |
| `decision_source` | `v6_pipeline` or `gemini_review`. Gemini is never the V6 numerical owner. |

The image API returns synchronous V6 centimetres through `/api/v1/estimate`. Worker/event execution is routed through `src/worker.py:process_camera_event()` and the shared event adapter. Missing, NaN, or infinite depth is unavailable; it must not be converted to 0 cm. A prediction completing successfully means only that inference ran. It does not verify that the estimated depth is accurate.

## 4. Evidence and correction policy

The shared evidence collector can provide water-mask coverage and spatial bands, mask metadata, YOLO object observations, semantic-classifier outputs, Depth Anything V2 relative-depth context, and available experimental region candidates. Availability depends on local artifacts and configuration; missing evidence remains unavailable. Depth Anything values are relative, not centimetres. Object overlap and contour estimates are diagnostic context, not measured depth.

Evidence collectors cannot set the primary or final depth. V6 records proposals and a correction trace with the original value, proposal, source, delta, acceptance state, reason, and evidence identifiers. The current policy is shadow/abstain: proposals are not accepted without a separately validated rule. No classifier, missing mask, relative-depth signal, object proxy, or legacy guard can silently overwrite V6 depth or force it to zero.

Do not switch `DIAGNOSTIC_ONLY` signals to numerical authority by changing an enum or configuration value. Promotion requires an explicit, evidence-backed policy and tests; there is currently no supported bypass switch.

## 5. Image output and saved reports

Default image output is intentionally concise. It shows the image filename, application depth, friendly source, correction status/reason, warnings only when their conditions occur, and the saved CSV path. It omits raw JSON, object dictionaries, bounding boxes, checkpoint hashes, repeated depth sections, and internal authority labels. The status line distinguishes completed inference from verified accuracy.

Example:

```text
Flood Depth Estimator — V6
Image: 01_test.jpg

Estimated depth: 74.35 cm
Depth source: EfficientNet
Correction: None — no validated alternative estimate
Inference: completed; estimate is not independently accuracy-verified.

Warnings:
- Scene classifiers disagree; review recommended.
- Gemini review unavailable: API key missing.

Report saved: reports\image_predictions\01_test_v6_prediction.csv
```

The CSV keeps the existing report schema, including the V6 values, application result, decision source, checkpoint metadata, trace identifiers, uncertainty/evidence, correction trace, Gemini review fields, and safe error diagnostics. A sibling `<image-stem>_v6_debug.json` preserves the complete serialized result and report. The JSON sidecar is written on each image run; `--debug` also prints the JSON and its path. `--verbose` prints a compact evidence summary. Per-image reports with the same filename stem are replaced on a later run.

Warnings are conditional. Examples include scene-classifier disagreement, unavailable V6 depth, or a Gemini key/authentication/quota/service/response problem. Gemini errors do not discard a valid V6 result. Invalid images and model/checkpoint loading failures remain explicit errors.

## 6. Gemini application reviewer

The reviewer is an optional post-inference layer implemented through the existing `src/llm_judge.py` and shared `review_v6_result()` boundary. It receives the original image, V6 metric values, genuine available diagnostics, and the controlled-correction trace. It does not run inside `V6ShadowPipeline.predict()` and cannot mutate `primary_depth_cm` or `final_shadow_depth_cm`.

Current non-secret settings in `config/config.yaml` are `inference.llm_judge.enabled: true`, `inference.llm_judge.apply_corrections: true`, model `gemini-flash-latest`, and `inference.video.gemini_review_enabled: false`. The relevant configuration shape is:

```yaml
inference:
  llm_judge:
    enabled: true
    apply_corrections: true
    # google_api_key is optional; prefer a secret manager or environment variable
  video:
    gemini_review_enabled: false
```

Check the file before changing behavior. Keep video review off unless per-frame API cost and latency have been deliberately accepted.

| `enabled` | `apply_corrections` | Behavior |
| --- | --- | --- |
| false | either | No Gemini request. Application final equals V6 final. |
| true | false | Advisory review only. The response is recorded; application final stays equal to V6 final. |
| true | true | A validated recommendation can change only application final when Gemini says the V6 prediction is incorrect. Otherwise V6 is retained. |

For local execution, provide the secret through the `GOOGLE_API_KEY` environment variable. Never paste the key into this handover, source code, logs, or a commit. A missing key, timeout, API failure, blocked or malformed response, schema failure, or invalid correction falls back to the V6 result and records a safe diagnostic. A proposed depth must parse as finite numeric centimetres, be non-negative, and stay within the configured physical maximum. Unit tests mock Gemini; they must not make real API calls.

## 7. Entrypoints and useful files

| File or route | Purpose |
| --- | --- |
| `main.py image` | Local/S3 image CLI orchestration, concise output, CSV and JSON report export. |
| `main.py video` | Video orchestration using saved JPEG frames and the shared image pipeline. |
| `web_app.py` | Flask UI, upload route `/predict`, estimate API, and health route. |
| `src/v6_inference.py` | Shared RGB decoding, pipeline factory, finite-depth handling, result serialization, and evidence collection. |
| `src/v6_shadow_pipeline.py` | V6 metric prediction contract, typed diagnostics, reliability notes, and sole controlled correction authority. |
| `src/v6_shadow_contract.py` | Typed V6 signals, evidence, correction trace, and result contracts. |
| `src/v6_application_review.py` | Shared post-inference Gemini orchestration and application-level result. |
| `src/llm_judge.py` | Existing Gemini API client and response parsing. |
| `src/pipeline.py` / `src/worker.py` | Shared event and worker/EC2 orchestration. |
| `scripts/run_v6_shadow_video.py` / `src/v6_video_input.py` | Saved-frame video input and V6 frame processing. |
| `src/v6_image_reporting.py` | Image report schema, CSV/JSON export, and CLI presentation. |
| `config/config.yaml` | Runtime checkpoint, evidence, Gemini, and video settings. |

The configured primary checkpoint is `models/candidate/best_flood_model_water_aware_hardneg.pth`. Treat the checkpoint file and configuration as a matched pair. Do not substitute weights or edit model transforms as part of UI/reporting work.

## 8. Validation and safe changes

Run the maintained test suite from the repository root:

```powershell
python -m pytest tests -q
```

For output-specific checks, run `python -m pytest tests/test_v6_console_output.py tests/test_v6_default_entrypoints.py -q`. The test suite covers concise output, conditional warnings, JSON detail preservation, invalid/unavailable values, and shared-entrypoint parity. Test output and generated image reports should stay in temporary directories; do not commit generated reports.

Safe presentation changes include layout, wording, colors, report formatting, and CLI presentation helpers. Keep the shared input contract, `V6ShadowPipeline.predict()`, model/checkpoint loading, transforms, result serialization, correction policy, and numerical ownership unchanged unless the task explicitly authorizes an inference change. Do not add a second UI/CLI/API/video prediction implementation.

The estimate is not a per-pixel depth map or surveyed water-level measurement. Current secondary evidence is not validated for numerical fusion. Accuracy work requires a fresh, frozen evaluation set; do not tune against already-consumed internal or external challenge data. Training and evaluation utilities are separate from normal inference.

## 9. Historical reference status

The frozen V4 implementation and the Legacy Production/Reference Pipeline (V5) remain in the repository for provenance and historical experiments. Residual Fusion Model (V14), old resolvers, guards, severity outputs, and other legacy stages do not participate in normal V6 prediction. Do not use the V5 path as a fallback or infer that retained code is active from its presence in the repository.

Older detailed experiment notes and training walkthroughs have been removed from this handover to avoid mixing historical results with the current runtime guide. Use the source files, configuration, and the specific experiment report being reviewed when those details are needed.
