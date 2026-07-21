# notebooks/ — Google Colab Training Notebooks

| Notebook | Purpose |
|----------|---------|
| `FloodWatch_Upgraded_Architecture.ipynb` | **Main** — EfficientNet-B4, multi-task training |
| `FloodWatch_MLOps_Training.ipynb` | Simpler fine-tune of existing B0 checkpoint |

## How to use (Upgraded Architecture)

1. Open in Colab: https://colab.research.google.com/github/Mishra-Kaumod/flood-depth-estimator/blob/main/notebooks/FloodWatch_Upgraded_Architecture.ipynb
2. Set `GEMINI_API_KEY` in Cell 1
3. Upload flood images to `/content/images/`
4. Run cells 1-12 in order
5. Download `floodnet_v2_results.zip` from Cell 12
6. Copy `best_floodnet_v2.pth` ? `models/` and commit

## Cell summary
| Cell | Action |
|------|--------|
| 1 | Config (set API key) |
| 2 | Create folders |
| 3 | Install packages |
| 4 | Load pre-trained weights |
| 5 | Gemini auto-label ? labels.csv |
| 6 | Split 400 train / rest test |
| 7 | Build FloodNetV2 |
| 8 | Augmentation + dataset |
| 9 | Training loop |
| 10 | Evaluate accuracy |
| 11 | Deep metrics (optional) |
| 12 | Download results ZIP |
