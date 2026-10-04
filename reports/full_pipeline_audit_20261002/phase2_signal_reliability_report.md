# Phase 2 Physical-Signal Reliability Analysis

## Scope And Limits

This is offline analysis only. It used the saved approved clean feature tables (465 train, 99 validation) and the eight Phase 1 live traces. No inference was rerun, no model was trained, and the consumed internal test and external challenge were not accessed.

The saved feature tables provide full-population metrics for EfficientNet, contour reference, region depth, current base, and the legacy dense proxy. Per-object YOLO diagnostics and mask-conditioned output exist only for the eight Phase 1 representative traces, so they are diagnostic evidence, not sufficient for ranking a model.

## Signal Reliability

| Signal | Clean validation availability | Validation MAE / bias | Assessment |
| --- | ---: | ---: | --- |
| EfficientNet depth | 99/99 | 9.17 cm / +2.50 cm | Strongest standalone physical centimetre signal. Retain, but it overestimates 0-10 cm by 18.41 cm MAE. |
| Contour `reference_depth_cm` | 99/99 returned values; YOLO refs 95/99 | 81.26 cm / +79.51 cm | Unsafe as a direct centimetre estimate. The value is from a separate contour path, not the YOLO references. |
| Region depth | 99/99 | 44.52 cm / +34.91 cm | Advisory/context only until recalibrated. Strong shallow-to-mid positive bias. |
| Current pre-residual base | 99/99 | 26.00 cm / +13.86 cm | Not reliable enough as an immutable physical anchor in its current form. |
| Depth Anything legacy proxy | 99/99 | Not evaluated as cm | Relative proxy rank correlation with actual depth is only 0.11 on validation. Do not interpret `p90 x 120` as metric depth. |
| Mask-conditioned candidate | 8/8 traces only | 11.27 cm / -0.69 cm | Insufficient evidence to rank; stored checkpoint validation R2 was 0.297. Advisory only. |

EfficientNet is best in 337/564 approved clean rows (59.75%) versus the current base. Examples where it is accurate but fusion/base is materially worse include `10cm_1.jpg` (7.7 vs actual 10.0; base 119.02) and `sept23_25cm_2.jpg` (27.09 vs actual 25.0; base 119.95).

## Depth-Bucket Pattern

EfficientNet is strongest from 20-75 cm on validation: 4.91 cm MAE in 20-30, 6.77 in 30-50, and 6.88 in 50-75. Its failure regime is shallow: 18.41 cm MAE in 0-10 and 11.66 in 10-20.

Contour reference remains severely high in every validation bucket: +113.71 cm bias at 0-10, +87.01 at 20-30, +72.46 at 30-50, and still +9.38 at 75+. More references do not repair it: validation reference-count versus reference-error rank correlation is **+0.375**; higher counts correlate with worse error in this data.

## YOLO/Submersion Diagnostics

The eight Phase 1 traces contain 30 detected objects: 15 cars, 10 people, 3 motorcycles, one bus, and one bicycle. The current per-object waterline proxy is explicitly diagnostic and is not suitable for direct depth:

- Cars: proxy MAE 86.64 cm, positive bias 72.64 cm.
- People: proxy MAE 87.28 cm, positive bias 58.27 cm.
- Bus/bicycle/motorcycle counts are too small for conclusions; the bus proxy error was 275.5 cm.

Useful future reliability features are still clear:

- detector confidence;
- bbox area/scene scale;
- submersion and waterline consistency across objects;
- proxy dispersion across objects;
- object class and vertical image position;
- disagreement with EfficientNet, region depth, and relative-depth proxy.

Examples:

- `image_4.jpg`: two small people have zero YOLO waterline proxies, yet separate contour reference depth is 108.9 cm. This confirms the current reference contract is inconsistent.
- `abhinav_20260923_16_4.5cm.jpeg`: eight objects span bus/car/person/motorcycle/bicycle and the diagnostic proxy range is 90-280 cm (190 cm dispersion) for a 4.5 cm label. This should be low reliability, not strong reference evidence.
- `25.04cm.png`: seven references have 126.41 cm proxy dispersion and low mean detector confidence (0.318), another low-reliability reference scene.

Across the saved 564-row feature table, maximum submersion has a +0.243 rank correlation with contour-reference absolute error. It is not a reliable quality score by itself.

## Relative Depth and Mask Candidate

Depth Anything is currently normalized per image. Its legacy numeric proxy has validation rank correlation 0.11 and Pearson correlation 0.005 with actual depth. Its spatial/region statistics may still be useful for mask geometry or disagreement detection, but it should not directly influence centimetres without separate calibration.

The mask-conditioned candidate is available only in the eight traces. Its apparent small-sample MAE is not enough to outweigh its stored validation R2 of 0.297, so it must remain advisory until independently validated on newly frozen data.

## Ranked Recommendation

1. **Retain as a physical centimetre candidate:** EfficientNet depth. It needs shallow-regime handling, not unrestricted authority.
2. **Retain as diagnostics only:** YOLO object detections, submersion, waterline, object type, bbox scale, and per-object dispersion. These are promising reliability features but not current depth estimators.
3. **Advisory/context only:** region depth and mask-conditioned candidate.
4. **Require separate calibration before centimetre use:** Depth Anything relative-depth statistics.
5. **Do not directly influence centimetres:** current contour `reference_depth_cm` and the current per-object waterline proxies. They may contribute only after the future reference-reliability contract proves them valid.
6. **Do not treat reference count as reliability:** count is currently positively associated with contour-reference error and is not tied to the contour depth calculation.

## Deferred

No fusion weight, reference contribution, cap, resolver, guard, checkpoint, or threshold was changed. Replacing the contour reference handoff, adding a reliability score to fusion, or creating a calibrated relative-depth conversion remains outside Phase 2.

Supporting artifacts:

- `phase2_signal_reliability_metrics.json`
- `phase2_train_validation_split_metrics.json`
- `phase2_per_image_yolo_diagnostic_summary.csv`
