**Segmentation Migration Plan**

Purpose
- Provide a clear, low-risk migration from heuristic/fallback water detection to a DeepLabV3-based semantic segmentation primary path while retaining legacy behavior and safety checks for flood-depth estimation.

1) Modules that become obsolete after DeepLabV3 integration
- **`water_detection.py`**: Primary role (handcrafted multi-method segmentation) becomes redundant once a robust DeepLabV3 model provides reliable pixel-wise water masks.
- **`final_water_detection_system.py`**: Its classifier + heuristic consensus should be retired as the primary segmentation source; it remains useful only for training-data augmentation or offline validation.
- TripleEnginePipeline.get_water_mask (in `core_logic.py`): Its placeholder routing to legacy classifier should be replaced by the segmentation engine.

2) Modules that remain useful as fallback / validators
- **`improved_water_detector.py`**: Keep as a real-time hallucination-prevention validator (object-visibility rules + simple checks). Run after segmentation to veto unlikely flood predictions.
- **`final_water_detection_system.py`**: Retain as an offline or low-cost fallback (CPU) for edge-cases and for datasets where segmentation fails; useful for model comparison and monitoring.
- **`water_detection.py`**: Keep a trimmed subset (select methods) inside a diagnostic/analysis module for dataset QA and visualization only.
- **`cv_engine.py`**: Keep as a compatibility wrapper for legacy callers and to mediate phased rollout.

3) Modules that should be merged
- Merge small, tightly-related validators into a single `validation` package:
  - `improved_water_detector.py` + selected safe checks from `water_detection.py` → `validators/hallucination_checks.py` (keeps object-visibility, color/edge heuristics, depth discontinuity checks).
  - Keep `final_water_detection_system.py` separate as `validators/consensus_fallback.py` (it loads classifiers and is heavier).

4) Modules that should remain independent
- **`core_logic.py`**: Keep independent — it implements depth estimation engines (A/B/C) and ensemble fusion; DeepLabV3 should feed masks/depth cues into this without inlining logic.
- **`cv_engine.py`**: Keep as the compatibility shim (single-file wrapper) until all call sites migrate.
- **Model serving layer**: Add a new independent `segmentation_engine.py` or `engines/segmentation.py` that encapsulates DeepLabV3 loading, preprocessing, and inference (exportable to TorchScript/ONNX).
- **`flood_api/tasks.py`** and temporal analyzer modules: remain independent and orchestrate distributed workers (Celery) calling segmentation + depth engines.

5) Recommended final production inference architecture
- Components (microservices or processes):
  - **Segmentation Service** (`segmentation_engine`): TorchScript/ONNX DeepLabV3 model, GPU-enabled, REST/gRPC or internal RPC, returns binary/multi-class water mask and per-pixel confidence.
  - **Depth Engine Service** (`core_logic`): Runs anchor detection (YOLO optionally converted to ONNX), Depth-Anything depth model (or an ONNX alternative) and ensemble fusion; isolated to worker nodes with GPUs.
  - **Validator Service** (`validators`): CPU-bound checks (`improved_water_detector` logic), cheap heuristics to prevent hallucinations; runs synchronously post-segmentation.
  - **Orchestrator / API** (`flood_api` Celery + web): Accepts uploads, enqueues segmentation + depth tasks, merges outputs, persists audit logs and produces final risk scoring.
  - **Storage & Metadata**: Object store (S3), Postgres for metadata + geotags, Redis for Celery; enable audit trails and model-version tagging.

6) Exact migration sequence (step-by-step)
- Phase 0 — Preparation
  1. Add `segmentation_engine.py` abstraction (no behavior change yet). Implement a lazy loader pattern to avoid import-time heavy loads.
  2. Add configuration flags (settings) to choose `SEGMENTATION_BACKEND: legacy|deeplab|noop` and `VALIDATOR_BACKEND`.

- Phase 1 — Data & model readiness
  1. Finalize dataset preprocessing (convert masks to target label schema per `DATASET_AUDIT_REPORT.md`).
  2. Train/validate DeepLabV3 (transfer-learning) on the prepared dataset; produce a validated checkpoint and export TorchScript/ONNX.

- Phase 2 — Local integration and feature flagging
  1. Implement `segmentation_engine` with two backends: `legacy` (calls `FinalWaterDetectionSystem` / `TripleEnginePipeline`) and `deeplab` (calls DeepLabV3). Default to `legacy`.
  2. Modify `TripleEnginePipeline.get_water_mask()` to call `segmentation_engine.get_mask()` instead of the internal placeholder.
  3. Keep `cv_engine.FloodDepthEngine.process_frame()` unchanged but ensure it reads segmentation via `segmentation_engine`.
  4. Add unit tests asserting backward-compatible output shapes and keys.

- Phase 3 — Canary rollout
  1. Enable `SEGMENTATION_BACKEND=deeplab` in a small canary environment; route 5–10% of traffic or a subset of test images.
  2. Run validators: post-process segmentation with `validators/hallucination_checks.py` and compare to `final_water_detection_system` outputs. Log disagreements to a monitoring bucket.
  3. Tune thresholds (mask confidence → water percent) and validators until false-positive rate is below target.

- Phase 4 — Gradual cutover and deprecation
  1. Increase traffic share for `deeplab` as canary metrics improve.
  2. When stable across locales (e.g., 2–4 weeks), flip default to `deeplab` and keep `legacy` as fallback for N% of requests or for low-confidence masks.
  3. Remove `water_detection.py` and `final_water_detection_system.py` from primary path; retain them behind a feature flag for offline analysis.

- Phase 5 — Hardening
  1. Convert models to optimized runtime (TorchScript/ONNX, INT8 quant where safe) and ensure reproducible CI model tests.
  2. Add observability: per-request features, model version, mask confidence histograms, and disagreement alerts.

7) What to retain for municipality-grade flood depth estimation
- Keep and harden `core_logic` ensemble (Engine A/B/C): preserves explainability (wheel geometry, anchor ratio, scene fallback).
- Keep YOLO anchor detection but move it to a batched/async worker to limit per-request latency; consider lightweight anchors-only models for production.
- Retain depth model (Depth-Anything) but evaluate replacing with an ONNX-exported, faster depth network or a monocular depth backbone tuned for street scenes.
- Retain uncertainty outputs (`engine_breakdown`, `ensemble_confidence`) and add timestamped audit logs, geolocation, and image metadata for municipal records.

8) What to retain for risk scoring and action recommendation
- Inputs required:
  - `estimated_depth_cm` + `ensemble_confidence` (from `core_logic`)
  - `water_coverage_pct` and polygonized mask area (from `segmentation_engine`)
  - `num_anchors_detected` and object counts (for exposed assets)
  - temporal trends (from `temporal_analysis.py`) — required for escalation
- Components to keep:
  - A `scoring` module that consumes the above signals and outputs tiered risk levels and recommended actions (e.g., road closure, alert residents).
  - Threshold tables and local calibration datasets (per municipality) stored in Postgres.

Operational notes & best practices
- Avoid heavy import-time model loads: implement lazy model loaders in `segmentation_engine` and `core_logic`.
- Prefer serving segmentation as an independent GPU-backed service; depth ensemble can run on a separate GPU or shared TPU/VM pool depending on throughput.
- Add regression tests comparing legacy outputs vs new pipeline on a held-out canonical image set.
- Maintain `cv_engine.py` as compatibility shim until all internal callers are migrated and validated.

Deliverables I can produce next (pick one)
- (A) `segmentation_engine.py` stub with lazy loader and feature-flag hooks.
- (B) Preprocessing pipeline script to convert the dataset to binary masks + produce train/val/test splits.
- (C) A small canary playbook and sample CI tests to validate model parity.

----
Generated by repository analysis — documentation only; no code edits were made to existing modules.
