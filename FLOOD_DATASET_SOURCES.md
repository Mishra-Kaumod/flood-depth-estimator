# Flood Dataset Sources

This document lists publicly accessible flood imagery datasets suitable for training DeepLabV3 models and municipal flood intelligence.

## 1. FloodNet

- **Official source**: FloodNet dataset, originally published for post-flood scene understanding.
- **Download URL**: https://huggingface.co/datasets/torchgeo/floodnet
- **Alternate source**: https://datasetninja.com/floodnet
- **License**: `cdla-permissive-1.0` on the Hugging Face rehosted dataset; verify original dataset license and attribution requirements when using Kaggle or official challenge archives.
- **Approximate size**: ~1–2 GB for the image/mask archive
- **Pixel masks included**: Yes. Includes pixel-level water segmentation masks and road/flood annotations.
- **Notes**: Hugging Face provides a public dataset card and download tooling compatible with `datasets` and `torchgeo`.

## 2. CVFD (Close-View Flood Dataset)

- **Official source**: Kaggle dataset page for the Close-View Flood Dataset (CVFD)
- **Download URL**: https://www.kaggle.com/datasets/sunnyshabanali/close-view-flood-dataset-cvfd
- **License**: Kaggle dataset terms apply; download requires a Kaggle account and acceptance of dataset rules.
- **Approximate size**: Likely several GB depending on image and mask compressions; exact archive size available on Kaggle.
- **Pixel masks included**: Yes. The dataset supports semantic segmentation and includes pixel-level annotations.
- **Notes**: This dataset is the most accessible publicly referenced CVFD release for flood image segmentation.

## 3. Sen1Floods11

- **Official source**: Sen1Floods11 dataset release
- **Download URL**: https://zenodo.org/records/5125603
- **License**: Open science research license; verify exact terms on Zenodo or the dataset landing page.
- **Approximate size**: ~20–30 GB for the full Sentinel-1 imagery plus mask archive
- **Pixel masks included**: Yes. Includes flood inundation masks aligned to SAR satellite imagery.
- **Notes**: Best for wide-area inundation segmentation and transfer learning from remote sensing to street-level tasks.

## 4. xBD (Building Damage Assessment)

- **Official source**: xView2 / xBD dataset release
- **Download URL**: https://xview2.org/
- **License**: Academic research use with attribution; check xView2 terms for commercial restrictions.
- **Approximate size**: Tens of GB for the full xBD dataset and high-resolution imagery
- **Pixel masks included**: Not direct water masks; contains building damage polygons that can be adapted for flood exposure or object-aware segmentation tasks.
- **Notes**: Useful for supplementing flood segmentation models with damage-aware building context.

## 5. DeepGlobe Flood Dataset

- **Official source**: DeepGlobe Flood challenge
- **Download URL**: https://competitions.codalab.org/competitions/18467
- **License**: Research/non-commercial competition license; verify all terms before use.
- **Approximate size**: ~2–5 GB
- **Pixel masks included**: Yes. Flood segmentation masks are available for satellite imagery.
- **Notes**: Good for satellite flood segmentation training and domain adaptation.

## Notes

- For municipal flood intelligence, prioritize datasets that include clear pixel-level water masks and street-level imagery.
- Always verify dataset license terms before commercial or operational use.
- For DeepLabV3 training, binary water masks are usually easiest to adopt; convert multi-class or palette masks to binary water/non-water labels when necessary.

## Recommended starting sources for DeepLabV3

1. FloodNet — best for street-level RGB images and existing mask pairs.
2. CVFD — strong candidate for close-view flood segmentation and mask-rich video frames.
3. Sen1Floods11 — best for wide-area inundation training and generalization.
4. DeepGlobe Flood — useful for satellite-based segmentation and transfer learning.
