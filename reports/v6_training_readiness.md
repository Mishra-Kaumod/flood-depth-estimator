# V6 Training Readiness

## Scope

This readiness infrastructure prepares future new-image ingestion, validation,
group-safe splitting, and reporting. It does not train a model, run inference,
tune thresholds, or modify V5/V6 numerical behavior.

## Manifest Contract

The header-only template is `templates/v6_new_image_manifest.csv`.

Required stable fields:

- `image_id`, `filename`, `sha256`
- `scene_type`, `depth_cm`, `label_confidence`, `measurement_source`
- `source_session_id`, `group_id`, `physical_reference`
- `depth_eligible`, `classification_only`, `quarantine_reason`
- `visual_near_duplicate_status`

Supported scene types are `DRY`, `WET_NO_FLOOD`, `FLOOD_0_10`,
`FLOOD_10_20`, `FLOOD_20_50`, `FLOOD_50_75`, and `FLOOD_75_PLUS`.

`DRY` and `WET_NO_FLOOD` require a blank `depth_cm`,
`classification_only=true`, and `depth_eligible=false`. Missing depth is never
converted to zero. Flood scenes require a positive depth and are quarantined if
they use `0 cm`, an invalid scene, invalid confidence/source, or inconsistent
eligibility flags.

Synthetic examples exist only in `tests/fixtures/`; the real-data template has
no rows.

## Duplicate and Grouping Contract

`scripts/prepare_v6_training_dataset.py` records three distinct checks:

1. Exact file duplicates by SHA-256.
2. Source/session grouping using `source_session_id`.
3. Visual near duplicates using dHash only when an explicit image root is
   supplied.

When image files are not supplied, visual status is written as `not_run`; it is
never reported as passed. Invalid/conflicting rows are written to quarantine and
are not silently corrected.

Duplicate/session/near-duplicate relationships are unioned into one deterministic
`group_id`, so a related group cannot cross split boundaries.

## Deterministic Splits

The preparation script creates `TRAIN`, `VALIDATION`, and `CHALLENGE` manifests
from group units using a persisted seed and split configuration. It writes:

- `split_config.json` with seed, fractions, row counts, and group counts;
- `duplicate_and_group_audit.json`;
- `leakage_check.json` for SHA and group intersections;
- `dataset_summary.json` by scene, depth bucket, confidence, source, and
  session;
- `challenge_freeze.json` with challenge image IDs, group IDs, SHA-256 hashes,
  split date/config, and challenge-manifest hash.

Future training/tuning must reject challenge rows. The included evaluation
template defaults to development mode and rejects `CHALLENGE` rows. A one-time
future Challenge evaluation requires both `--mode final_evaluation` and an
existing `--candidate-freeze` artifact. This is an explicit gate for a frozen
candidate, not a training/tuning option.

Dataset summary generation is deliberately integrated into
`scripts/prepare_v6_training_dataset.py`; no separate summary script is needed.
It writes `dataset_summary.json` with per-split row/group counts and counts by
scene type, depth bucket, label confidence, measurement source, and source
session. The same command also writes split configuration, duplicate/group audit,
leakage audit, quarantine, and Challenge freeze artifacts.

## Future Training and Evaluation Placeholders

`configs/v6_training_skeleton.yaml` contains no runnable training command. It
documents blocked placeholders for future shallow/regime refinement and optional
refinement components.

`scripts/evaluate_v6_future_template.py` consumes precomputed predictions only
and reports V5, V6 primary, and future V6 refined MAE/RMSE, depth buckets
`0-10`, `10-20`, `20-50`, `50-75`, `75+`, severe under/over counts, and the
two catastrophic-underestimate metrics.

## Tests Run

```text
python -m py_compile src/v6_training_data_contract.py \
  scripts/prepare_v6_training_dataset.py \
  scripts/evaluate_v6_future_template.py \
  tests/test_v6_training_data_contract.py

python -m unittest discover -s tests -p "test_v6_training_data_contract.py" -v
```

Result: 5 synthetic-only tests passed.

The tests cover classification-only dry handling, invalid scene quarantine,
zero-depth flood quarantine, deterministic group-safe split generation with
visual status `not_run`, challenge freeze generation, and challenge-row rejection
by the future evaluation template.

## Blocked Until New Images Exist

- Real manifest validation and SHA/visual duplicate audit.
- Final group-safe TRAIN/VALIDATION/CHALLENGE creation.
- Human review of quarantined labels and source/session metadata.
- Any model training, refinement training, calibration, or final evaluation.

## Final Synthetic End-to-End Smoke Test

Only `tests/fixtures/v6_end_to_end_synthetic.csv` was used. It contains an
intentional exact SHA-256 duplicate pair and one invalid `0 cm` flood label.
Two independent runs with seed `42`, validation fraction `0.20`, and Challenge
fraction `0.20` produced identical TRAIN, VALIDATION, CHALLENGE, and Challenge
freeze artifact hashes. The invalid flood row was quarantined; the duplicate pair
was assigned to one group; SHA/group leakage was empty; and visual duplicate
status was `not_run` because no image root was supplied.

The smoke test exercised manifest validation, quarantine, exact-duplicate
grouping, deterministic group-safe split generation, leakage checks, Challenge
freeze creation, and dataset summary creation. No real images, production files,
inference, training, threshold tuning, internal test, or external Challenge data
were accessed.
