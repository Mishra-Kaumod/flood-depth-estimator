# Flood Detection & Depth Estimation System

A comprehensive Python system for detecting flood presence, classifying severity, and estimating water depth from images and videos.

## Features

✅ **Water Detection** - Advanced multi-method water surface detection  
✅ **Severity Classification** - ResNet18-based flood severity (0-4 levels)  
✅ **Depth Estimation** - Maps severity to estimated water depth  
✅ **Single Image Analysis** - Quick analysis of individual images  
✅ **Video Processing** - Frame-by-frame analysis with CSV export  
✅ **YOLO Object Detection** - Vehicles and people detection for anchor-based depth  
✅ **AWS S3 Integration** - Cloud storage for inputs and results  
✅ **GPU Support** - CUDA acceleration when available  

## Storage Modes

- **Local Mode** (DEFAULT) - Read/write files from local directories
- **AWS S3 Mode** - Read/write files directly from S3 bucket

For S3 setup instructions, see [S3_SETUP.md](S3_SETUP.md)

## Project Structure

```
flood_project_cleaned/
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
├── severity_model.pth         # Trained ResNet18 model (must be placed here)
├── .env.example               # AWS credentials template
├── README.md                  # This file
├── S3_SETUP.md                # AWS S3 integration guide
├── modules/
│   ├── __init__.py
│   ├── water_detection.py     # WaterDetectionAnalyzer class (6 methods)
│   ├── predict_image.py       # SeverityPredictor class (ResNet18)
│   ├── process_video.py       # VideoFloodAnalyzer class (video analysis)
│   ├── depth_band_estimator.py # Depth severity mapping
│   ├── object_detection.py    # ObjectDetector class (YOLO)
│   ├── hybrid_depth_estimator.py # HybridDepthEstimator (3-method ensemble)
│   └── s3_handler.py          # S3Handler class (AWS S3 I/O)
└── test_images/              # Sample test images
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. **Clone/Extract the project**
   ```bash
   cd flood_project_cleaned
   ```

2. **Create virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add trained model**
   - Place `severity_model.pth` in project root directory
   - This file contains the trained ResNet18 model weights

## Usage

### 1. Analyze Single Image

```bash
# Local storage (DEFAULT)
python main.py image test_images/sample.jpg

# AWS S3 storage
python main.py image images/sample.jpg --storage=aws

# Output:
# FLOOD SEVERITY PREDICTION
# ============================================================
# Image: test_images/sample.jpg
# Severity: 2 - Moderate Flood
# Confidence: 85.32%
# Depth Band: 20-50 cm
# Estimated Depth: 35 cm
# 
# All Probabilities:
#   No / Very Low Flood: 0.0234
#   Minor Flood: 0.1456
#   Moderate Flood: 0.8532
#   High Flood: 0.0678
#   Severe Flood: 0.0100
```

### 2. Process Video

```bash
# Local storage (DEFAULT)
python main.py video sample_video.mp4

# Optional: Custom output CSV and frame skip rate
python main.py video sample_video.mp4 results.csv 2

# AWS S3 storage
python main.py video videos/sample.mp4 results.csv 1 --storage=aws

# Arguments:
#   video_path: Path to video file
#   output_csv: Name of results CSV (default: video_analysis.csv)
#   skip_frames: Process every Nth frame (default: 1, process all)
#   --storage: Local or AWS (default: local)
```

### 3. Object Detection (NEW)

```bash
# Local storage
python main.py object test_images/sample.jpg annotated.jpg

# AWS S3 storage
python main.py object images/sample.jpg objects_output.jpg --storage=aws
```

**Output files:**
- `video_analysis.csv` - Frame-by-frame analysis results
- `output_frames/frame_*.jpg` - Annotated frames (every 5 frames)
- `output_frames/output_video.mp4` - Annotated video
- `objects_detected.jpg` - Image with object bounding boxes

## System Architecture

### Pipeline Flow

```
Input (Image or Video Frame)
    ↓
1. WATER DETECTION (WaterDetectionAnalyzer)
   └─ Uses 6 computer vision methods:
      • Color-based analysis (HSV)
      • Edge detection (Canny + Hough)
      • Contrast pattern analysis
      • Horizontal surface line detection
      • Depth discontinuity detection
      • Ripple/motion pattern detection
   └─ Consensus voting (≥3 methods required)
    ↓
2. SEVERITY CLASSIFICATION (only if water detected)
   └─ ResNet18 neural network
   └─ 5 output classes (0-4)
    ↓
3. DEPTH ESTIMATION
   └─ Map severity class to depth band
   └─ Output: depth_cm (5-100 cm)
    ↓
Output (Detection results + depth estimate)
```

## Core Modules

### WaterDetectionAnalyzer
**Location**: `modules/water_detection.py`

Detects water presence using multiple techniques:
```python
from modules import WaterDetectionAnalyzer
import cv2

detector = WaterDetectionAnalyzer()
image = cv2.imread("image.jpg")
result = detector.detect_water_surface(image)

print(f"Water detected: {result['water_detected']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Water coverage: {result['water_percentage']:.1%}")
```

### SeverityPredictor
**Location**: `modules/predict_image.py`

Classifies flood severity:
```python
from modules import SeverityPredictor

predictor = SeverityPredictor(model_path="severity_model.pth")
result = predictor.predict("image.jpg")

print(f"Severity: {result['severity_name']}")
print(f"Depth: {result['depth_cm']} cm")
```

### VideoFloodAnalyzer
**Location**: `modules/process_video.py`

Processes video files:
```python
from modules import VideoFloodAnalyzer

analyzer = VideoFloodAnalyzer(model_path="severity_model.pth")
df = analyzer.process_video("video.mp4", output_csv="results.csv", skip_frames=1)

print(f"Frames analyzed: {len(df)}")
print(f"Avg water %: {df['water_percentage'].mean():.2f}%")
```

## Output Formats

### Single Image Output
```
Image: test_images/sample.jpg
Severity: 2 - Moderate Flood
Confidence: 85.32%
Depth Band: 20-50 cm
Estimated Depth: 35 cm
```

### Video CSV Output
```
frame_number, time_seconds, water_detected, water_confidence, water_percentage, 
severity_class, severity_name, severity_confidence, depth_band, depth_cm

0, 0.0, True, 0.9, 45.23, 2, "Moderate Flood", 0.85, "20-50 cm", 35
1, 0.033, True, 0.88, 44.56, 2, "Moderate Flood", 0.83, "20-50 cm", 35
2, 0.066, False, 0.1, 2.34, None, "N/A", None, "N/A", None
...
```

## Severity Classes

| Class | Name | Depth Band | Depth (cm) |
|-------|------|------------|-----------|
| 0 | No / Very Low Flood | 0-5 cm | 5 |
| 1 | Minor Flood | 5-20 cm | 15 |
| 2 | Moderate Flood | 20-50 cm | 35 |
| 3 | High Flood | 50-80 cm | 65 |
| 4 | Severe Flood | 80+ cm | 100 |

## Performance Notes

- **Water Detection**: ~50-100ms per frame
- **Severity Classification**: ~200-500ms per frame (depends on GPU)
- **Video Processing**: ~1-2 FPS with full pipeline
- **GPU**: 2-3x faster with CUDA

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install -r requirements.txt
```

### "Model file not found"
- Ensure `severity_model.pth` is in project root
- Check file exists: `ls severity_model.pth`

### "CUDA out of memory"
- Reduce video resolution
- Process every Nth frame: `python main.py video file.mp4 results.csv 2`

### Slow performance
- Ensure GPU acceleration: Check "Using device: cuda" message
- Install CUDA toolkit if available
- Use `skip_frames` to process fewer frames

## Development

### Testing single module
```python
# Test water detection
python -m modules.water_detection

# Test image prediction
python modules/predict_image.py test_images/sample.jpg

# Test video processing
python modules/process_video.py sample_video.mp4
```

## Model Requirements

The system requires a trained ResNet18 model (`severity_model.pth`):
- **Architecture**: ResNet18 with 5 output classes
- **Input**: 224x224 RGB images
- **Output**: 5-class probability distribution
- **Training Data**: Flood severity dataset

## License

This project is provided as-is for research and development purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model file is present
4. Check file paths are correct

## Version

- **Version**: 1.0.0
- **Last Updated**: 2026-06-20
- **Python**: 3.8+
