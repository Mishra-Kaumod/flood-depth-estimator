# Dataset Audit Report — dataset_labeled

Summary
- **Images:** 290 (JPEG)
- **Masks:** 290 (PNG, single-channel grayscale)
- **Metadata pairs:** 290 (dataset_labeled/metadata.csv) — all pairs present, no missing files

Mask characteristics
- All masks are single-channel (grayscale). None are strictly binary; every mask contains multiple unique label values.
- Union of unique pixel values across the dataset: 256 distinct values (0–255 seen).
- Typical number of distinct values per mask (sample): ~29–43 unique values.
- This indicates masks are not simple binary water/non-water labels. They likely encode graded labels (e.g., depth levels) or use a palette with many indices.

Spatial consistency
- Most image/mask pairs match in dimensions, but **7** pairs have mismatched shapes (examples):
  - 14.jpg ↔ 14.png (image (1425,1900) vs mask (630,1024))
  - 15.jpg ↔ 15.png (image (630,1024) vs mask (1425,1900))
  - 2052.jpg ↔ 2052.png (image (490,1012) vs mask (800,1200))
  - 2053.jpg ↔ 2053.png (image (800,1200) vs mask (490,1012))
  - 3059.jpg ↔ 3059.png (image (400,650) vs mask (1005,1920))
  - 1061.jpg ↔ 1061.png (image (440,660) vs mask (389,700))
  - 1079.jpg ↔ 1079.png (image (682,1024) vs mask (422,759))

Quality concerns & implications
- Non-binary masks: If your training target is binary water segmentation, these masks must be converted (e.g., map all non-zero values to a single `water` class). Without conversion, semantic segmentation models will try to predict many classes (up to 256), which is likely not intended.
- Mixed relative scales: The 7 mismatched pairs will cause label misalignment during training — these must be fixed (by re-resizing, re-labeling, or excluding)
- Image size heterogeneity: Images/masks use multiple resolutions (examples: (720,1280), (682,1024), (800,1200), etc.). Choose a consistent input size or implement on-the-fly resizing in the data pipeline.

Recommendations (next steps)
- Confirm label semantics: ask the dataset author whether mask values encode class indices, depth graduations, or color palette indices. This determines preprocessing.
- If target is binary water mask:
  - Convert masks to binary (mask_bin = (mask > 0).astype(uint8)).
  - Re-save as single-channel PNG with values {0,255} or {0,1}.
- If mask values encode depth/continuous labels:
  - Decide whether to treat task as regression (depth estimation) or discretize into n classes for segmentation. DeepLabV3 supports multi-class segmentation but requires a clear class mapping.
- Fix mismatched pairs: either resize masks to image size (preferred: perform resizing using nearest-neighbor for masks) or exclude/relabel the problematic pairs.
- Normalize dataset sizes: choose a canonical training resolution (e.g., 512x512 or 720x1280) and implement resizing + data augmentation.
- Augmentation & validation: with only 290 images, use heavier augmentation and consider k-fold cross-validation.

Recommended split (example)
- Given size = 290, a balanced option: **70% train / 20% val / 10% test** → ~203 / 58 / 29 images.
- Alternative (if you need more validation): 80/10/10 → 232 / 29 / 29.

Suitability for DeepLabV3 baseline
- Feasible for a baseline fine-tune if:
  - Masks are preprocessed into the intended label scheme (binary or a well-defined set of classes).
  - Dimension mismatches are fixed and images resized to a consistent input size.
  - You use transfer learning (pretrained backbone) and strong augmentation because the dataset is small.
- Not sufficient for robust, production-ready segmentation without additional labeled examples or strong augmentation and careful validation.

Quick scripts & checks
- List mismatched pairs (already executed): see examples above.
- Convert masks to binary (nearest-neighbor when resizing). Example snippet:

```python
import cv2, os
from glob import glob
for png in glob('dataset_labeled/Mask/*.png'):
    m=cv2.imread(png, cv2.IMREAD_UNCHANGED)
    binm=(m>0).astype('uint8')*255
    cv2.imwrite(png.replace('/Mask/','/Mask_bin/'), binm)
```

Action items I can do next (pick one)
- (A) Produce a scripted preprocessing pipeline that: validates pairs, fixes mismatched sizes (nearest-neighbor), converts masks to binary or re-indexed classes, and writes train/val/test CSV splits.
- (B) Visualize a random sample of 20 image/mask pairs and their unique mask values to confirm label semantics.
- (C) Start a small DeepLabV3 fine-tuning notebook using this dataset (after you choose binary vs multi-class).

If you'd like, I can implement (A) to produce consistent, trainer-ready dataset artifacts and the split CSVs.

----
Report generated automatically by repository audit tools.
