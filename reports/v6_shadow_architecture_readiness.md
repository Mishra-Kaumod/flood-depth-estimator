# V6 Shadow Architecture Readiness

## Scope

V6 is a development-only shadow architecture on branch
`flood-model-v6-shadow-dev`. It does not alter V5 code, configuration,
thresholds, checkpoints, or prediction behavior. No training, tuning, internal
test, or external-challenge evaluation was performed.

## Files Created

- `src/v6_shadow_contract.py`
- `src/v6_shadow_pipeline.py`
- `src/v6_shadow_comparison.py`
- `scripts/run_v6_shadow_comparison.py`
- `tests/test_v6_shadow_contract.py`
- `tests/test_v6_shadow_pipeline.py`

## Numerical Ownership

The initial V6 shadow numerical contract is intentionally simple:

```text
primary_depth_cm = EfficientNet candidate depth
final_shadow_depth_cm = primary_depth_cm
owner = efficientnet_primary_anchor
```

No other current signal can alter V6 centimetres. The optional refinement
interface is explicitly a no-op. V14 is not connected as a V6 refinement or
final decision owner.

## Signal Categories

| Category | Current V6 signals | Authority |
| --- | --- | --- |
| Metric depth | EfficientNet candidate | PRIMARY_METRIC |
| Semantic/context | Road-scene, no-water, wet-road, mask coverage | CONTEXT_ONLY |
| Object/reference | YOLO object data, confidence, size, position, submersion, waterline, diagnostic proxy, contour estimate | DIAGNOSTIC_ONLY |
| Relative depth | Depth Anything p90/min/max | CONTEXT_ONLY, unit `relative` |
| Advisory | Region estimate, mask-conditioned candidate | ADVISORY_ONLY |

Depth Anything has no V6 `_cm` signal. Its legacy V5 conversion is not reused.
Contour reference depth and nominal-height YOLO proxies are never numerical
owners in V6.

## Reliability and Uncertainty

V6 records, without changing depth:

- missing or malformed signals;
- backend fallback activity;
- raw EfficientNet-to-context differences;
- object proxy median and dispersion when valid objects are available;
- semantic and object disagreement status.

No probability, coverage, object count, dispersion, guard, or uncertainty flag
currently caps, zeroes, replaces, or corrects V6 depth.

## Stage Contract

Immutable snapshots are emitted for input, segmentation/context, semantic
outputs, EfficientNet primary depth, object diagnostics, relative-depth
diagnostics, reliability, optional-refinement input/output, uncertainty, and
final shadow output. Each snapshot records its numerical before/after state and
owner.

## Comparison Support

`scripts/run_v6_shadow_comparison.py` compares one explicitly approved image
only after SHA-256 verification. It reports V5 final depth, V6 primary/final
shadow depth, their difference, reliability/uncertainty information, and stage
snapshots. It does not enumerate a dataset or evaluation directory.

## Tests Run

```text
python -m py_compile src/v6_shadow_contract.py src/v6_shadow_pipeline.py \
  src/v6_shadow_comparison.py scripts/run_v6_shadow_comparison.py \
  tests/test_v6_shadow_contract.py tests/test_v6_shadow_pipeline.py

python -m unittest discover -s tests -p "test_v6_shadow_*.py" -v
```

Result: 10 tests passed.

The tests verify missing-value preservation, NaN rejection, string-boolean
rejection, relative-depth naming, V5 payload immutability, one V6 numerical
owner, and isolation from extreme contour (`500`), YOLO proxy (`300`), legacy
dense (`120`), and semantic-probability (`0.99`) inputs.

## Integration Smoke Test

Three SHA-256-verified images from the frozen clean validation manifest were
run once for integration only. This was not an accuracy evaluation and did not
use internal test or external challenge data.

| Image | Actual cm | V5 final cm | V6 EfficientNet primary cm | V6 final shadow cm | V6 owner | Object disagreement | Backend fallback |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `066_20CM.jpg` | 20.0 | 18.44 | 7.68 | 7.68 | `efficientnet_primary_anchor` | available | false |
| `030_35CM.jpg` | 35.0 | 35.84 | 23.38 | 23.38 | `efficientnet_primary_anchor` | available | false |
| `image_4.jpg` | 35.0 | 0.00 | 22.95 | 22.95 | `efficientnet_primary_anchor` | available | false |

For all three runs, `V6 final_shadow_depth_cm == V6 primary_depth_cm` exactly.
Semantic disagreement remains `unassessed_no_calibrated_policy` and is
reporting-only. In particular, `image_4.jpg` confirms that a V5 final zero does
not regain V6 ownership: V6 retained the EfficientNet primary depth of 22.95
cm through every snapshot.

The corresponding immutable comparison outputs are stored under
`reports/v6_shadow_smoke/`. They are engineering smoke artifacts, not a new
evaluation set or tuning input.

## Blocked Until New Data / Training

- A calibrated shallow/no-flood refinement model.
- A trained reliability or uncertainty model.
- A calibrated metric conversion for relative Depth Anything values.
- Evidence-based reference reliability weighting.
- Any production promotion or numerical refinement policy.

V6 is ready for controlled shadow engineering comparisons, not production use.
