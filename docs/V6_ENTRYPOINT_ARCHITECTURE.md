# Shared V6 inference architecture

Audited on 2026-10-08 on flood-model-v6-shadow-dev.

All active V6 inference entrypoints converge as follows:

UI / image CLI / video CLI (including execution on EC2)
→ src.v6_inference.load_v6_rgb
→ src.v6_inference.create_v6_pipeline
→ V6ShadowPipeline.predict(image_rgb)
→ final_shadow_depth_cm

| Entrypoint | Input orchestration | Prediction/output |
| --- | --- | --- |
| web_app.py POST /predict | Uploaded encoded bytes, shared PIL RGB loader; lazily reused pipeline | Shared factory, predict(), shared final cm payload |
| scripts/run_v6_shadow_comparison.py | Explicit image path and SHA-256 verification, shared RGB loader | Shared factory, predict(), result-owned comparison and stages |
| scripts/run_v6_shadow_video.py | Existing decoder, save JPEG frame, reload through shared RGB loader | Shared factory, predict() for each saved frame, shared final cm payload |

No separate V6 EC2 inference runner or V6 API was found. Running these entrypoints
on EC2 uses exactly the same implementation. main.py and the camera/temporal API
routes remain V5; they are not V6 entrypoints. Future V6 API/deployment adapters
must use src.v6_inference rather than construct their own models or preprocessors.

The shared input contract is encoded image bytes or a file decoded by PIL to
uint8 H×W×3 RGB. It performs no resizing or normalization. Model preprocessing
remains in the existing underlying model implementation. Video retains JPEG
encoding; parity with image entrypoints is defined for the saved JPEG frame,
not for the pre-encoding frame array.

Only V6ShadowPipeline owns V6 numerical prediction. Adapters contain no depth
thresholds, corrections, fusion, resolver, or averaging. The shared finite-depth
serializer preserves the prior video's invalid/nonfinite handling; it does not
change a valid V6 result. Comparison formatting remains owned by the existing
V6 result class. Model logic, defaults, checkpoints, training, and V5 are unchanged.

Validation: tests/test_v6_entrypoint_parity.py exercises the actual Flask route
and both CLI main functions with the real V6ShadowPipeline and controlled V5
signals. A channel-sensitive synthetic frame is saved with the existing video
JPEG flow, then used through every entrypoint. Fractional cm, zero and missing
depth agree exactly. Bytes/file RGB loading also agrees for grayscale and RGBA.
This checks adapter parity without loading trained checkpoints or downloading
models; trained-checkpoint inference is not part of this test.

## Real trained-checkpoint parity smoke

On 2026-10-08, the previously approved 066_20CM.jpg (SHA-256
f35019bbd4bc8d9721a035d9a7d817937a1113ef64940a0acda1269db7eb30fa)
was run through the actual image CLI and Flask upload route with real models.
Both primary_depth_cm and final_shadow_depth_cm were 7.68 in both paths.
The first frame of the previously tested DAV was saved and reloaded through
process_saved_frames; it returned 77.39 for both fields. The independent image
CLI on that exact saved JPEG also returned 77.39 for both fields. Each pair
had identical decoded RGB SHA-256 values and exactly 0.0 cm differences.

Every path verified a loaded, eval-mode EfficientNet primary checkpoint at
models/candidate/best_flood_model_water_aware_hardneg.pth, SHA-256
78bd165a30b8fee67d17d7ec13b840d6cbdd9d2504445443bf61d52248b5fb40.
Instrumentation recorded the real predict() result without replacing inference.
No accuracy evaluation, training, or model changes were performed. Local smoke
outputs are under reports/v6_entrypoint_parity_smoke and are not part of this
refactor commit.
