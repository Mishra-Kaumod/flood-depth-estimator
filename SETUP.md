# Quick Start Guide

## For Windows Users

### Step 1: Install Python & Virtual Environment
```bash
# Navigate to project
cd flood_project_cleaned

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Model File
```bash
# Check if severity_model.pth exists
dir severity_model.pth
```

Should show: `severity_model.pth` (size: ~45-50 MB)

### Step 4: Test with Sample Image
```bash
# List available test images
dir test_images

# Analyze a test image
python main.py image test_images\image_1.jpg
```

### Step 5: Process a Video (Optional)
```bash
python main.py video sample_video.mp4 results.csv 1
```

## For Linux/macOS Users

### Step 1: Setup
```bash
cd flood_project_cleaned
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Model
```bash
ls -lh severity_model.pth
```

### Step 4: Test
```bash
python3 main.py image test_images/image_1.jpg
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install -r requirements.txt
# or
pip install torch==2.2.1 torchvision==0.17.1
```

### Issue: "severity_model.pth not found"
**Solution:**
- Ensure model file is in project root
- Check with: `ls severity_model.pth` (Linux/macOS) or `dir severity_model.pth` (Windows)

### Issue: "No module named 'PIL'"
**Solution:**
```bash
pip install Pillow
```

### Issue: Slow processing (no GPU)
- The system uses CPU if CUDA is not available
- Processing will be slower but still functional
- To install CUDA support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### Issue: "CUDA out of memory"
**Solution:**
```bash
# Process every 2nd frame instead
python main.py video large_video.mp4 results.csv 2
```

## File Locations Reference

```
flood_project_cleaned/
├── main.py                    ← Run this
├── severity_model.pth         ← Must exist (trained model)
├── requirements.txt           ← Dependencies list
├── README.md                  ← Documentation
├── SETUP.md                   ← This file
├── modules/
│   ├── water_detection.py     ← Water detection engine
│   ├── predict_image.py       ← Image classification
│   ├── process_video.py       ← Video processing
│   └── depth_band_estimator.py ← Depth mapping
└── test_images/               ← Sample images (100+ provided)
```

## Common Commands

```bash
# Analyze single image
python main.py image test_images\image_50.jpg

# Process video with default settings
python main.py video my_video.mp4

# Process video, save every frame
python main.py video my_video.mp4 output.csv 1

# Process video, skip frames for speed
python main.py video my_video.mp4 output.csv 3

# Show help
python main.py
```

## Output Locations

After processing:

**Single Image:**
- Output appears in console

**Video:**
- `video_analysis.csv` - Results table
- `output_frames/` folder:
  - `frame_*.jpg` - Annotated frames
  - `output_video.mp4` - Annotated video

## Next Steps

1. ✅ Verify installation works
2. ✅ Test with provided sample images
3. ✅ Try video processing with small test video
4. ✅ Integrate into your application

## Support

If you encounter issues:
1. Check Python version: `python --version` (should be 3.8+)
2. Verify dependencies: `pip list | grep torch`
3. Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
4. See README.md for detailed troubleshooting
