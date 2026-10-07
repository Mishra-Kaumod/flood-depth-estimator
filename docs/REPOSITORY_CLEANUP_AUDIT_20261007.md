# Repository cleanup review — 2026-10-07

Branch: `flood-model-v6-shadow-dev`. This pass organizes files only. No source, config, checkpoint, threshold, training logic, model behavior, or dataset path changed. Nothing was deleted permanently, staged, committed, or pushed.

## 1. Current folder-tree summary

```text
flood_project_cleaned/
├── main.py, web_app.py, evaluate_model_readiness.py
├── MASTER_KT_HANDOVER.md, MASTER_KT_HANDOVER.docx
├── README.md, SETUP.md, S3_SETUP.md, S3_QUICKSTART.md
├── src/                 runtime, training and experimental source modules
├── scripts/             training, evaluation, audit and experimental CLIs
├── tests/               V6 image/video/data contracts and CSV fixtures
├── config/              active V5 configuration and candidate training config
├── configs/             V6 future-training skeleton
├── models/              active, fallback and historical weights
│   └── candidate/       includes several ACTIVE weights despite the name
├── reports/             current evidence and historical experiments
├── docs/                training, architecture, readiness and this review
├── templates/           web templates and V6 manifest template
├── data/                app database, manifests, quarantine and generated output
├── training_data/, evaluation_data/, flood_dataset/, test_images/
├── training_runs/       local images, split manifests and experiment outputs
├── archive/
│   ├── legacy_cli/      still referenced by runtime/tests; preserve paths
│   ├── 20260921_legacy_training_assets/
│   └── reports/legacy_timestamped/   newly archived 27 reports
├── archive_cleanup/     existing local recovery material; ignored
├── generated_docs/      generated document work; ignored
├── __agent__/           local document QA/cleanup work; now ignored
└── .venv/, .pytest_cache/, __pycache__/    local environments/caches
```

## 2. Proposed cleaned structure

```text
src/          scripts/       tests/        templates/
config/       configs/       models/       docs/
reports/      # approved current V5/V6, provenance and accepted comparison evidence
archive/
  reports/legacy_timestamped/     # implemented in this pass
  experiments/                   # future only, after reference verification
  scripts/                       # future only, after reference verification
  deprecated_models/             # future only, after workflow verification
  historical/                    # future only, after KT/reference verification
```

Keep both `config/` and `configs/`: they have different existing consumers. Do not rename them for appearance. Keep local datasets and training-run directories where current manifests/scripts expect them. `archive/legacy_cli` cannot be assumed unused: existing Python imports depend on it. A future source migration is outside this cleanup.

## 3. Classification and ACTIVE files/folders

| Scope | Classification | Action / rationale |
|---|---|---|
| `main.py`, `web_app.py`, `src/pipeline.py`, `src/segformer_yolo_depthv2_pipeline.py`, runtime dependencies in `src/` | ACTIVE | Keep entry points and imported modules in place. |
| `src/v6_shadow_contract.py`, `src/v6_shadow_pipeline.py`, `src/v6_shadow_comparison.py`, `src/v6_video_input.py` | ACTIVE | Current V6 shadow image/video path; no separate API added. |
| `scripts/run_v6_shadow_comparison.py`, `scripts/run_v6_shadow_video.py`, `scripts/prepare_v6_training_dataset.py`, `scripts/evaluate_v6_future_template.py` | ACTIVE | V6 CLI/data-contract/future evaluation infrastructure; future training remains gated. |
| `evaluate_model_readiness.py`, `scripts/audit_full_pipeline_trace.py`, current training/evaluation/data preparation scripts | ACTIVE or REFERENCE / KEEP | Preserve all code: historical defaults and evaluation consumers still reference old versions. |
| Remaining experimental `src/` and `scripts/` files | REFERENCE / KEEP or UNKNOWN / NEED REVIEW | No code moved. Filename age alone does not establish safe removal. Rejected routers remain inactive. |
| `tests/`, root import/web/S3 test files | ACTIVE / legacy validation reference | Preserve tests and fixtures. Root S3 test uses obsolete `modules/` imports; not run as an active V5 readiness check. |
| `config/`, `configs/`, `requirements.txt`, `.env.example`, `templates/` | ACTIVE | Preserve exact paths and content. |
| Eight V5 registered checkpoints plus `yolov8n.pt` | ACTIVE | V6 still invokes V5 extraction. See complete checkpoint map below. |
| Other checkpoints, root `severity_model.pth` | REFERENCE / KEEP or UNKNOWN / NEED REVIEW | No weight moves; historical/manual workflow usage not fully proven. |
| Master KT Markdown/Word, README, setup/S3 docs, training guide, residual training guide, architecture/readiness docs | ACTIVE / REFERENCE / KEEP | Preserve all documentation and links. No archived report is referenced by MASTER KT. |
| `reports/flood_model_readiness_guard_v5_manifest.json`, `reports/architecture_investigation_freeze_20261002.json`, `reports/full_pipeline_audit_20261002/` | REFERENCE / KEEP | Frozen V5 identity, parity, trace and signal-reliability evidence. |
| `reports/v6_shadow_architecture_readiness.md`, `reports/v6_training_readiness.md`, `reports/v6_shadow_smoke/`, readiness smoke folders | ACTIVE / REFERENCE / KEEP | Preserve immutable engineering/readiness evidence. |
| `reports/` data-audit, provenance, baseline, V14/V15, collapse/safety/router/two-stage experiment folders and feature/evaluation CSVs | REFERENCE / KEEP | Supporting evidence or referenced inputs. Rejected does not mean disposable. Keep at current paths until documentation/evaluator consumers are migrated together. |
| 27 timestamped readiness/comparison reports listed below | ARCHIVE CANDIDATE → ARCHIVED | Historical pre-V5 output; no literal references found; preserve original filenames and bytes. |
| `training_data/`, `evaluation_data/`, `flood_dataset/`, `test_images/`, training-run manifests | ACTIVE / REFERENCE / KEEP / UNKNOWN | No data moved. Consumed internal-test/challenge data remain reporting-only, never tuning inputs. |
| `training_runs/` image copies, weights, caches and raw training outputs | GENERATED / IGNORE with embedded reproducibility references | Existing ignore retained; do not discard split manifests/provenance merely because the parent is ignored. |
| `.venv/`, Python/pytest caches, logs, `temp/`, `generated_docs/`, `__agent__/`, extracted frames and video CSV/run outputs | GENERATED / IGNORE | Leave locally; ignore targeted outputs. |
| `archive_cleanup/`, existing historical archives, unspecified local files | REFERENCE / KEEP or UNKNOWN / NEED REVIEW | Leave untouched; recovery/import/path dependencies may exist. |

## 4. Audit method and archived files

Scanned 622 text files across source/imports, configs, model loaders, tests, CLI/shell/PowerShell files, README/MASTER KT, deployment/configuration material, training/evaluation scripts, report/manifests and historical recovery code. Search included basename occurrences, including absolute Windows path strings. Files were read up to 20 MB; binary checkpoint contents and Word XML were not treated as a reference corpus. The Word master is generated from the retained Markdown. No code/model/dataset was moved on the strength of a negative text search.

Also checked generic report-name producers: `evaluate_model_readiness.py` writes timestamped `model_readiness_*.json`; `scripts/compare_mask_conditioned_fusion.py` writes timestamped comparison CSVs. Their discovery loops enumerate input images, not these historical output reports. Selected reports predate the frozen V5 evidence; they remain available for historical comparison under archive.

All 27 moved files were untracked and previously hidden by `reports/`. Native PowerShell `Move-Item -LiteralPath` was used after checking absolute source/destination containment and destination absence. SHA-256 was checked across every move. No `git mv` was needed because none was tracked. No active import/path/config needed updating.

Destination for every file below: `archive/reports/legacy_timestamped/`.

- `mask_conditioned_fusion_comparison_20260911_170743.csv`
- `mask_conditioned_fusion_comparison_20260911_171450.csv`
- `mask_conditioned_fusion_comparison_20260911_172251.csv`
- `mask_conditioned_fusion_comparison_20260911_173911.csv`
- `model_readiness_20260824_232932.json`
- `model_readiness_20260824_233121.json`
- `model_readiness_20260824_233440.json`
- `model_readiness_20260824_234001.json`
- `model_readiness_20260909_002501.json`
- `model_readiness_20260909_002741.json`
- `model_readiness_20260911_144754.json`
- `model_readiness_20260911_145123.json`
- `model_readiness_20260911_151935.json`
- `model_readiness_20260911_152646.json`
- `model_readiness_20260911_153722.json`
- `model_readiness_20260911_162503.json`
- `model_readiness_20260911_165559.json`
- `model_readiness_20260919_144932.json`
- `model_readiness_20260919_145403.json`
- `model_readiness_20260919_145753.json`
- `model_readiness_20260919_150726.json`
- `model_readiness_20260919_233430.json`
- `model_readiness_20260919_233505.json`
- `model_readiness_20260923_002858.json`
- `model_readiness_20260923_003108.json`
- `model_readiness_20260923_003355.json`
- `model_readiness_20260923_003835.json`

## 5. Generated files / .gitignore

Removed blanket `reports/` ignore so approved reports and manifests can be reviewed for version control. Added targeted rules:

```gitignore
reports/v6_video_runs/
reports/v6_dav_smoke_*.csv
reports/**/frames/
reports/**/*training*.log
reports/**/*.err.log
__agent__/
logs/
temp/
*.log
*.DAV
*.dav
*.DHAV
*.dhav
training_data/**/*.jpg
training_data/**/*.jpeg
training_data/**/*.png
training_data/**/*.webp
training_data/**/*.avif
evaluation_data/**/*.jpg
evaluation_data/**/*.jpeg
evaluation_data/**/*.png
evaluation_data/**/*.webp
evaluation_data/**/*.avif
flood_dataset/**/*.jpg
flood_dataset/**/*.jpeg
flood_dataset/**/*.png
```

Existing environment, Python/pytest caches, candidate weight, training-run, local dataset-folder, database and generated-document ignores remain. New raw-image rules leave CSV/JSON labels/manifests visible. Ignore rules do not untrack files already committed and do not undo existing dataset deletions. No generated files or datasets were staged. Approved historical traces and engineering smoke JSONs remain visible because they are reproducibility evidence, not disposable temporary output. Newly visible historical CSV/report files require explicit selection before any later commit; do not add the entire reports directory blindly.

The 27 previously untracked historical timestamped reports remain local only. `/archive/reports/legacy_timestamped/` is ignored specifically; current `reports/` evidence and other archive directories are not hidden by this rule. An individual historical report should enter Git only after its reproducibility or active-documentation requirement is specifically reviewed.

## 6. Items deliberately left untouched

All 52 checkpoint files inventoried, every source/script, all dataset folders and training-run split/provenance files, rejected experiments referenced by KT/reports, existing archive/recovery directories, master Word/Markdown, and all pre-existing working-tree modifications/deletions. Model files with no literal reference remain UNKNOWN, not automatically obsolete. Historical script defaults are still useful reproduction contracts. This is a conservative cleanup, not a claim that every remaining historical file is required by live inference.

The active wet-road checkpoint is still ignored/untracked as already recorded in the V5 manifest. A self-contained release needs a separately reviewed model distribution/tracking decision; this pass did not force-add or change checkpoint policy.

## 7. Checkpoint usage map

Full evidence, including exact reference filenames, is in [REPOSITORY_CHECKPOINT_USAGE_20261007.json](REPOSITORY_CHECKPOINT_USAGE_20261007.json). Reference evidence is not necessarily a loader: training output defaults and historical comparison paths are distinguished from the verified active loader list. All checkpoints stayed in place.

| Checkpoint | Loaded by / code-reference evidence | Status |
|---|---|---|
| `models/best_flood_model_water_aware.pth` | src/pipeline.py::_load_weights (configured alternate/fallback path); src/train_water_aware.py | ACTIVE configured V5/V6 dependency |
| `models/candidate/best_flood_model_test0916.pth` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/best_flood_model_water_aware_hardneg.pth` | SegformerYoloDepthV2Pipeline::_load_efficientnet_signal_if_available; V6 primary anchor | ACTIVE configured V5/V6 dependency |
| `models/candidate/depth_regime_classifier_v1.pt` | scripts/train_depth_regime_classifier.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/depth_regime_classifier_v2_packaged.pt` | src/segformer_yolo_depthv2_pipeline.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/depth_regime_classifier_v3_finetuned.pt` | scripts/train_depth_regime_classifier_finetune.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/depth_regime_classifier_v4_balanced.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/depth_regime_classifier_v5_targeted_hardneg.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/depth_regime_classifier_v6_targeted_conservative.pt` | SegformerYoloDepthV2Pipeline::_load_depth_regime_classifier_if_available | ACTIVE configured V5/V6 dependency |
| `models/candidate/dev_indian_traffic_depth/control_seed123.pt` | scripts/compare_targeted3_depth_seeds.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/dev_indian_traffic_depth/control_seed17.pt` | scripts/compare_targeted3_depth_seeds.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/dev_indian_traffic_depth/residual_v14_clean_control.pt` | scripts/audit_depth_augmentation_effect.py; scripts/compare_depth_dev_control_vs_augmentation.py; scripts/compare_targeted3_depth_seeds.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/dev_indian_traffic_depth/residual_v14_data_aug.pt` | scripts/compare_depth_dev_control_vs_augmentation.py; scripts/compare_v14_vs_candidate_validation.py; scripts/run_residual_fusion_depth_dev.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/dev_indian_traffic_depth/targeted3_seed123.pt` | scripts/compare_targeted3_depth_seeds.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/dev_indian_traffic_depth/targeted3_seed17.pt` | scripts/compare_targeted3_depth_seeds.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/dev_indian_traffic_depth/targeted3_seed42.pt` | scripts/compare_targeted3_depth_seeds.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/flood_depth_expert_v1.pt` | scripts/evaluate_hierarchical_depth_router.py; scripts/train_flood_depth_expert.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/flood_depth_expert_v2_deep_weighted.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/flood_depth_expert_v3_safety_biased.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/flood_depth_expert_v4_no_leak_safety.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/fusion_depth_model_atharva.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/fusion_depth_model_combined_atharva.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/no_water_guard_teammate_augmented.pth` | SegformerYoloDepthV2Pipeline::_load_no_water_guard_if_available | ACTIVE configured V5/V6 dependency |
| `models/candidate/residual_fusion_depth_model_ankle_v10.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_ankle_v11.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_ankle_v12_clean.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_ankle_v13_clean_anchor.pt` | scripts/evaluate_hierarchical_depth_router.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/residual_fusion_depth_model_ankle_v14_no_leak.pt` | SegformerYoloDepthV2Pipeline::_load_residual_fusion_if_available | ACTIVE configured V5/V6 dependency |
| `models/candidate/residual_fusion_depth_model_ankle_v2.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_ankle_v3.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_ankle_v4.pt` | reports/broad_mask_cap_audit/reproduce_audit.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/residual_fusion_depth_model_ankle_v5.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_ankle_v6.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_ankle_v7_uncapped_weighted.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_ankle_v8_uncapped_weighted.pt` | scripts/train_residual_regime_router.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/residual_fusion_depth_model_ankle_v9a.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_ankle_v9b.pt` | scripts/evaluate_residual_safety_fallback.py; scripts/train_residual_regime_router.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/residual_fusion_depth_model_ankle_v9c.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_retrain_old_eff_20260816_231536.pt` | scripts/train_residual_regime_router.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/residual_fusion_depth_model_scene_aware_v1.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_depth_model_v15_candidate.pt` | scripts/compare_v14_v15_candidates.py; scripts/run_two_stage_development_experiment.py | REJECTED / HISTORICAL; preserve reproduction |
| `models/candidate/residual_fusion_regime_router_v1.pt` | scripts/train_residual_regime_router.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/residual_fusion_regime_router_v2_objects.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/residual_fusion_regime_router_v3_waterline.pt` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/road_scene_classifier_4class_shallow_v2.pth` | SegformerYoloDepthV2Pipeline::_load_road_scene_classifier_if_available | ACTIVE configured V5/V6 dependency |
| `models/candidate/scene_guard_3class_test0916.pth` | scripts/train_scene_guard.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `models/candidate/wet_road_no_water_guard_augmented.pth` | No literal code/config reference found; external/manual usage unknown | UNKNOWN / NEED REVIEW: no proven active use; no move |
| `models/candidate/wet_road_no_water_guard_test0916.pth` | SegformerYoloDepthV2Pipeline::_load_no_water_guard_if_available (secondary guard) | ACTIVE configured V5/V6 dependency |
| `models/FloodDepth-MaskConditionedFusion.pth` | SegformerYoloDepthV2Pipeline::_load_mask_conditioned_fusion_if_available | ACTIVE configured V5/V6 dependency |
| `models/no_water_guard_mobilenet_v3_small.pth` | scripts/train_no_water_guard.py; src/segformer_yolo_depthv2_pipeline.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `severity_model.pth` | web_app.py; archive/legacy_main.py; archive/legacy_cli/modules/flood_analyzer.py; archive/legacy_cli/modules/predict_image.py; archive/legacy_cli/modules/process_video.py; archive/legacy_cli/modules/production_pipeline.py | REFERENCE / KEEP: historical/default training/evaluation or legacy reference; not selected in active config |
| `yolov8n.pt` | SegformerYoloDepthV2Pipeline::_load_yolo_if_available / ObjectDetector | ACTIVE object detector |


All eight active checkpoint hashes match `reports/flood_model_readiness_guard_v5_manifest.json`. `best_flood_model_water_aware.pth` is the configured alternate/fallback/training checkpoint; the current fusion mode uses the hard-negative EfficientNet weight for its depth signal. YOLO and cached Hugging Face Depth Anything were active in this smoke; depth teachers stayed disabled as configured.

## 8. Validation results

| Check | Result |
|---|---|
| Python syntax/compile (in-memory; no new caches required) | 117 Python files passed |
| Source/config/active checkpoint content hashes | All 121 recorded files unchanged |
| Existing V5 legacy import check (`test_imports.py`) | Passed |
| Existing V5 web route/home-page tests | 2 passed; expensive legacy black-image test intentionally omitted |
| V6 image/shadow/signal/data/video-input unittest discovery | 20 passed before and after cleanup |
| V6 saved-frame video regression | 1 passed before and after cleanup |
| Eight active checkpoint identity hashes | Match frozen V5 manifest |
| One V5 real-image smoke | 0.00 cm before and after |
| Same real-image V6 shadow smoke | Primary/final 22.95 cm before and after |
| Capped V6 video smoke (one decoded MP4 frame) | Primary/final 25.85 cm before and after; OpenCV; processed successfully |
| Comparison/backend payload | Exact JSON equality before and after |

Smoke input: `training_runs/fusion_ankle_depth_v8/images/train/image_4.jpg`, an image already used in the approved clean-validation engineering smoke. SHA-256: `79b3f47a80e8c4c48101c4889345c444e760294b052edafef3cbfbd4dfda8149`. Historical source directory names are not split authority. No internal-test/external-challenge dataset was enumerated for inference or tuning. Video was synthesized from that image and capped at one frame, so its different depth reflects video/JPEG encoding, not cleanup.

The first online initialization stalled and was stopped before inference; successful before/after runs both used `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` with the existing cache. Depth Anything loaded successfully; no dense-depth fallback occurred. These checks establish parity for this smoke scope, not accuracy improvement or exhaustive V5 parity. No models were trained.

Commands used for existing test suites:

```powershell
.venv/Scripts/python.exe test_imports.py
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_v6*.py'
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_run_v6_shadow_video.py'
```

Before/after smoke payloads are preserved locally as `__agent__/repo_cleanup_20261007/before_smoke.json` and `after_smoke.json`. Local helpers and raw status snapshots remain ignored under `__agent__/repo_cleanup_20261007/`; they are not project runtime dependencies. The optional untracked smoke export is excluded from the cleanup staging scope.

## 9. Git status and staged diff

At completion of the original cleanup pass, no changes were staged. No commit or push. The working tree was already dirty with substantial code/document changes, many deleted tracked images and untracked development files; this pass preserved them. The removal of blanket report ignores makes historical evidence visible as untracked files. Full local status snapshots are `__agent__/repo_cleanup_20261007/status_before.txt` and `status_after.txt`.

Status counts at review generation (two-character porcelain status):

- ` D`: 372
- ` M`: 9
- `??`: 233

Cleanup edits are `.gitignore`, the 27 untracked report moves, and these review/evidence files. Do not use `git add .`; select reviewed documentation/report paths individually and exclude pre-existing dataset deletions, raw data, model binaries, caches and generated outputs from a documentation/cleanup commit.

### Isolated staging review — 2026-10-08

The final authorized cleanup staging scope is exactly three paths: `.gitignore`, this audit document, and `docs/REPOSITORY_CHECKPOINT_USAGE_20261007.json`. The 27 historical reports were briefly staged for review, then unstaged at the user's request to avoid adding previously untracked historical output. They remain locally intact and ignored. The optional smoke export and archive README are excluded. No runtime/KT path updates are needed. The optional smoke-export link was replaced with the existing local evidence paths so the staged documentation does not depend on an excluded new artifact.

All 372 tracked deletions match the pre-cleanup status snapshot and are excluded. The eight other tracked modifications also predate cleanup and are excluded. Git's tracked-file list and history show that the 27 reports were not previously tracked; their local move history is recorded above, but no report addition or rename is included in the final index.

A repeat reference check covered 162 documentation/manifest files, including the master KT Markdown and Word XML, root documentation, and Markdown/JSON under `docs/` and `reports/`. None explicitly references any of these 27 reports outside this cleanup inventory. The inventory records local archive history and does not make the reports active-documentation or reproducibility dependencies. No historical report needs a tracking exception based on that check.
