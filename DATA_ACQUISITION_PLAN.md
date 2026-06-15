**DATA ACQUISITION PLAN — Municipality Flood Intelligence (V1)**

Purpose
- Define trusted sources, acquisition steps, storage sizing, licensing checks, and recommended V1 dataset stack for a municipal flood intelligence platform.

1) Trusted flood segmentation datasets
- **FloodNet**: street-level RGB images with water segmentation masks (good for street/urban scenes).
- **Sen1Floods11 / Sen2Floods**: Sentinel SAR/optical flood inundation datasets — good for wide-area inundation detection (coarse spatial resolution).
- **xBD (building damage)**: building-level damage labels including flooded/damaged buildings — useful for object-level risk labeling and training building-aware masks.
- **OpenStreetMap-derived flood annotations**: use OSM building/road footprints to validate masks and derive exposure labels.

2) Trusted flood depth datasets (sources & proxies)
- **LiDAR + survey flood event labels (USGS / local municipal LiDAR)**: align pre-/post-event DEMs with inundation extents to compute depth (best source for ground truth depth).
- **FEMA / national inundation maps + local topography (DEM/SRTM/ALOS)**: combine flood extents with high-resolution DEM to derive approximate depths.
- **Field measurement datasets (municipal sensor logs, camera + gauge pairs)**: high-value but sparse; use for calibration.
- **Synthetic / simulated depth data**: CFD or synthetic scene rendering for edge cases when real depth GT is missing.

3) FloodNet acquisition steps (street-level imagery)
- Research & licensing: confirm FloodNet license (usually research/academic) and citation requirements.
- Download the dataset (official release or mirror) and verify checksums.
- Validate metadata: ensure image↔mask pairing and coordinate/time metadata if present.
- Preprocess: resize/normalize, fix mismatched mask sizes, convert masks to intended schema (binary or multi-class), and store canonical filenames.
- Augment: simulate varied lighting, occlusions, and small rotations for better generalization.

4) CVFD acquisition steps (community video/frames dataset)
- Identify official CVFD source or contact maintainers; obtain dataset and license terms.
- Extract frames from video at a fixed rate (e.g., 1 fps) and pair with provided masks/annotations.
- Enrich with temporal metadata (frame index, timestamp) to enable temporal models.
- Validate and sample-verify masks for labeling consistency.

5) Licensing constraints & compliance checklist
- Always check dataset license before redistribution: common terms include **CC-BY**, **CC-BY-NC**, **ODbL**, or research-only restrictions.
- Commercial vs non-commercial: if the municipality product is commercial (paid service), avoid NC-licensed data or obtain commercial licenses.
- Attribution: record required attribution and include text in product docs and metadata records.
- Sensitive infrastructure: mask or avoid publishing datasets that reveal critical infrastructure locations unless cleared.
- Data residency/privacy: follow local rules for personally identifiable imagery (faces/plates) — apply blurring/anonymization where required.

6) Storage estimates (rough)
- Street images: assume ~1.0 MB per RGB image (1024×768 jpeg) + mask 0.05–0.2 MB → ~1.2 MB per paired sample.
  - 10k samples → ~12 GB; 100k → ~120 GB.
- Satellite tiles (Sentinel): ~10–50 MB per tile; Sen1Floods11 (SAR) average ~20 MB.
  - 1k tiles → ~20 GB.
- LiDAR / DEM: ~100s MB to several GB per tile depending on region/resolution.
- Metadata, indices, and backups: budget additional 20–30% overhead.
- Practical guidance: start with 1 TB storage to cover mixed data (images, masks, DEMs) for a city-scale pilot.

7) Recommended V1 dataset stack (minimum viable)
- Core (pixel-level segmentation): FloodNet + CVFD (street-level) — canonicalized, mask-validated, binary water label.
- Wide-area (inundation): Sen1Floods11 (SAR) + optional Sentinel-2 optical tiles for corroboration.
- Depth calibration: small curated LiDAR/DEM + municipal gauge records for a subset of locations (derive depth GT where possible).
- Derived metadata: OSM footprints, weather/time, device/camera metadata, geolocation (lat/lon), and event id.
- Suggested splits: 70% train / 20% val / 10% test; reserve a separate municipal holdout region for final acceptance tests.

Operational recommendations
- Maintain provenance: store dataset manifests with source URL, license, checksum, and ingest date.
- Use a reproducible preprocessing pipeline (Docker + script) that: validates pairs, fixes mask sizes (nearest-neighbor), converts to binary/multi-class, and writes TFRecord/COCO/Cityscapes-style outputs.
- Automate checks: missing pairs, size mismatches, unique mask values, and small-sample visual validation.
- Plan for iterative labeling: use semi-supervised model predictions to queue human review for low-confidence areas.

Next steps I can implement
- (A) A preprocessing script that canonicalizes masks to binary and writes train/val/test CSVs.
- (B) A small manifest generator that records source, license, checksum for each dataset asset.

Generated by repository analysis tools; I can start (A) or (B) on request.
