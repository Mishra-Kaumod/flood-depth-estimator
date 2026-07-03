# FLOOD DETECTION & DEPTH ESTIMATION SYSTEM
## Complete End-to-End Flow Analysis

---

## 📋 EXECUTIVE SUMMARY

This is a comprehensive **Flood Detection and Water Depth Estimation System** that analyzes images and videos to:
1. **Detect water presence** using 6 computer vision methods
2. **Classify flood severity** using ResNet18 deep learning (5 levels: 0-4)
3. **Estimate water depth** using hybrid methods combining severity class, object anchors, and image analysis
4. **Detect objects** (vehicles, people) for context-aware analysis and depth anchoring
5. **Process video streams** frame-by-frame with CSV export
6. **Support cloud storage** via AWS S3

---

## 🏗️ SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│          FLOOD DETECTION & DEPTH ESTIMATION v2.1            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT LAYER                                                 │
│  ├─ Local Files (test_images/, videos/)                     │
│  └─ AWS S3 Bucket (cloud storage)                           │
│       ↓                                                       │
│  STORAGE ROUTER                                             │
│  ├─ S3Handler (AWS S3 operations)                           │
│  └─ Local file I/O (direct filesystem)                      │
│       ↓                                                       │
│  PROCESSING PIPELINE (3 Modes)                              │
│  ├─ IMAGE MODE:    Single image analysis                    │
│  ├─ VIDEO MODE:    Frame-by-frame processing                │
│  └─ OBJECT MODE:   YOLO object detection + depth           │
│       ↓                                                       │
│  CORE ANALYSIS MODULES                                      │
│  ├─ 1. WaterDetectionAnalyzer (6 methods)                   │
│  ├─ 2. SeverityPredictor (ResNet18)                         │
│  ├─ 3. HybridDepthEstimator (3 methods)                     │
│  └─ 4. ObjectDetector (YOLOv8)                              │
│       ↓                                                       │
│  OUTPUT LAYER                                                │
│  ├─ Console output (real-time results)                      │
│  ├─ CSV files (video analysis)                              │
│  ├─ Annotated images (bounding boxes)                       │
│  └─ Annotated video (frame sequence)                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 DETAILED PIPELINE FLOW

### **FLOW 1: SINGLE IMAGE ANALYSIS** (`python main.py image <path>`)

```
Input Image (JPG/PNG)
    ↓
[WATER DETECTION] - WaterDetectionAnalyzer.detect_water_surface()
    │
    ├─ Method 1: Color-based analysis (HSV color space)
    │            Identifies water blue/cyan hues
    │
    ├─ Method 2: Edge detection (Canny + Hough Lines)
    │            Detects water surface edges
    │
    ├─ Method 3: Contrast analysis
    │            Identifies high-contrast water regions
    │
    ├─ Method 4: Horizontal surface line detection
    │            Finds water horizon lines
    │
    ├─ Method 5: Depth discontinuity (if depth map available)
    │            Identifies water-ground boundaries
    │
    ├─ Method 6: Ripple/motion pattern detection
    │            Detects water ripple textures
    │
    ├─ Consensus Voting: ≥3 methods must agree
    └─ Output: water_detected (bool), confidence, percentage
    ↓
DECISION: Is water present?
    │
    ├─ NO  → Output: "No Flood Detected" (SKIP severity classification)
    │
    └─ YES → Continue to next stage
    ↓
[SEVERITY CLASSIFICATION] - SeverityPredictor.predict_bgr()
    │
    ├─ ResNet18 deep learning model
    ├─ Input: 224×224 RGB image tensor
    ├─ Output: 5-class probability distribution
    │    Class 0: No/Very Low Flood (0-5 cm depth)
    │    Class 1: Minor Flood (5-20 cm)
    │    Class 2: Moderate Flood (20-50 cm) ← Most common
    │    Class 3: High Flood (50-80 cm)
    │    Class 4: Severe Flood (80+ cm)
    │
    └─ Output: severity_class, confidence, all_probabilities
    ↓
[DEPTH ESTIMATION] - HybridDepthEstimator.estimate_depth()
    │
    ├─ Method 1: Severity-based mapping (fallback)
    │    Maps class 0-4 to depth ranges: 5, 15, 35, 65, 100 cm
    │
    ├─ Method 2: YOLO object anchor-based
    │    Uses detected objects (people, vehicles) as size references
    │    Calculates water level relative to object waterline
    │
    ├─ Method 3: Hybrid ensemble
    │    Combines severity, water percentage, and object-based estimates
    │    Weighted averaging for final depth
    │
    └─ Output: depth_cm, depth_band, method_used, detailed_breakdown
    ↓
[OUTPUT]
    └─ Console display with formatted results:
       • Image path
       • Water detection status
       • Severity level (0-4)
       • Estimated depth in cm
       • Flood level description (Very Low, Minor, Moderate, etc.)
```

### **FLOW 2: VIDEO ANALYSIS** (`python main.py video <path> [output.csv] [skip_frames]`)

```
Video File (MP4/AVI)
    ↓
[FRAME EXTRACTION] - cv2.VideoCapture()
    │
    ├─ Read frame-by-frame
    ├─ Apply frame skip: process every Nth frame (skip_frames parameter)
    ├─ Calculate timestamps: frame_number * (1000/fps) milliseconds
    └─ Maintain frame counter
    ↓
FOR EACH FRAME:
    │
    └─ RUN COMPLETE PIPELINE (same as Single Image Analysis above)
       │
       ├─ Water detection
       ├─ Severity classification (if water detected)
       ├─ Depth estimation
       │
       └─ Store results in DataFrame row:
          frame_number, time_seconds, water_detected, water_confidence,
          water_percentage, severity_class, severity_name,
          severity_confidence, depth_band, depth_cm
    ↓
[FRAME VISUALIZATION] - Every 5 frames:
    │
    ├─ Draw water mask overlay (semi-transparent)
    ├─ Add text annotations with results
    ├─ Save annotated frame: output_frames/frame_XXXX.jpg
    ├─ Collect frames into video: output_frames/output_video.mp4
    │
    └─ Optionally embed annotations on frames
    ↓
[CSV EXPORT] - pandas.DataFrame.to_csv()
    │
    ├─ Save detailed frame-by-frame results
    ├─ Columns: 10 data fields + metadata
    ├─ Format: CSV with headers
    │
    └─ File: video_analysis.csv (or custom name)
    ↓
[STATISTICS] - Console summary:
    │
    ├─ Total frames processed
    ├─ Frames with water detected (count & percentage)
    ├─ Average water percentage across all frames
    ├─ Severity distribution (histogram)
    ├─ Average depth when water detected
    │
    └─ Display on console
    ↓
[UPLOAD] - If AWS S3 mode:
    │
    ├─ Upload CSV to S3 bucket
    ├─ Upload annotated frames to S3 (optional)
    ├─ Clean up temporary local files
    │
    └─ Confirm with console message
```

### **FLOW 3: OBJECT DETECTION** (`python main.py object <image_path> [output.jpg]`)

```
Input Image
    ↓
[YOLO DETECTION] - ObjectDetector.detect_objects()
    │
    ├─ YOLOv8 nano model (real-time detection)
    ├─ Detects: person, car, truck, bus, motorcycle, bicycle, etc.
    ├─ Output: class name, confidence, bounding box coordinates
    │
    └─ For each detection:
       • Class: person, vehicle type
       • Confidence score (0-100%)
       • Bounding box: (x1, y1, x2, y2) pixel coordinates
       • Dimensions: width × height in pixels
       • Area: width × height (pixel area)
    ↓
[OBJECT INVENTORY] - ObjectDetector.create_object_inventory()
    │
    ├─ Count objects by class
    ├─ Calculate average confidence per class
    ├─ Group detections by type
    │
    └─ Output: Object counts and statistics
    ↓
[ANCHOR-BASED DEPTH ESTIMATION] - ObjectDetector.estimate_depth_from_object()
    │
    ├─ Get largest detected object (best depth reference)
    ├─ Known physical dimensions used:
    │    • Person: 170 cm height, 45 cm width
    │    • Car: 145 cm height, 175 cm width
    │    • Truck: 200 cm height, 250 cm width
    │
    ├─ Calculate perspective ratio:
    │    pixels_to_cm = object_height_cm / bbox_height_pixels
    │
    ├─ Estimate water level at object waterline
    ├─ Convert pixel distance to physical distance
    │
    └─ Output: estimated_depth_cm, reference_object, method
    ↓
[VISUALIZATION] - ObjectDetector.draw_detections()
    │
    ├─ Draw bounding boxes for each detection
    ├─ Label with class name and confidence
    ├─ Color-code by object class
    │
    └─ Annotated image
    ↓
[OUTPUT]
    │
    ├─ Console: Detection results and depth estimate
    ├─ Image: objects_detected.jpg (or custom path)
    │    with bounding boxes and labels
    │
    └─ Optional S3 upload if cloud mode
```

---

## 📦 CORE MODULES BREAKDOWN

### **1. WaterDetectionAnalyzer** (`modules/water_detection.py`)
**Purpose**: Detect water presence using ensemble of 6 methods

**6 Detection Methods**:
1. **Color-based (HSV)**: Identifies blue/cyan water hues
   - HSV ranges: 90-130° (blue spectrum)
   - High saturation/value filtering
   
2. **Edge Detection (Canny + Hough)**: Finds sharp boundaries
   - Canny edge detection at threshold 50-150
   - Hough transform for line detection
   - Water surface = horizontal lines
   
3. **Contrast Analysis**: Detects contrast patterns
   - Compares local vs global contrast
   - Water = high contrast regions
   
4. **Horizontal Line Detection**: Finds water horizon
   - Morph operations isolate horizontal edges
   - Identifies water-sky boundary
   
5. **Depth Discontinuity**: Uses depth maps (if available)
   - Sharp depth changes indicate water boundary
   - Requires optional depth input
   
6. **Ripple/Motion Patterns**: Detects water texture
   - Template matching for ripple patterns
   - Optical flow analysis

**Consensus Voting**: ≥3 methods must agree for positive detection
- Reduces false positives
- Improves confidence score

**Output**:
```python
{
    'water_detected': bool,           # Final decision
    'confidence': 0.0-1.0,            # How confident (0-100%)
    'water_percentage': 0.0-100.0,    # % of image that is water
    'water_mask': ndarray,            # Binary mask of water
    'method_votes': {                 # Individual method results
        'rgb_color_analysis': bool,
        'edge_detection': bool,
        'contrast_analysis': bool,
        'horizontal_line_detection': bool,
        'depth_discontinuity': bool,
        'optical_flow_ripples': bool
    }
}
```

### **2. SeverityPredictor** (`modules/predict_image.py`)
**Purpose**: Classify flood severity using ResNet18 deep learning

**Model Architecture**:
- **Base**: ResNet18 (18-layer convolutional neural network)
- **Training Data**: Trained on flood severity dataset
- **Input**: 224×224 RGB images
- **Output**: 5-class probability distribution
- **Classes**: 0 (No/Very Low), 1 (Minor), 2 (Moderate), 3 (High), 4 (Severe)

**Processing Pipeline**:
1. Convert BGR image to RGB
2. Resize to 224×224 pixels
3. Normalize: subtract ImageNet mean, divide by std
4. Pass through ResNet18
5. Apply softmax to get probability distribution
6. Get argmax for class prediction

**Output**:
```python
{
    'severity_class': 0-4,            # Predicted class
    'severity_name': str,             # 'Moderate Flood', etc.
    'confidence': 0.0-1.0,            # Class confidence
    'all_probabilities': {            # All 5 class probabilities
        0: 0.05,
        1: 0.15,
        2: 0.75,  # highest
        3: 0.04,
        4: 0.01
    }
}
```

### **3. HybridDepthEstimator** (`modules/hybrid_depth_estimator.py`)
**Purpose**: Estimate water depth using 3 combined methods

**Method 1: Severity-based (Fallback)**
- Maps severity class to depth range:
  - Class 0 → 5 cm
  - Class 1 → 15 cm
  - Class 2 → 35 cm
  - Class 3 → 65 cm
  - Class 4 → 100 cm

**Method 2: Object Anchor-based (Primary)**
- Uses detected objects (people, vehicles) as height references
- Known dimensions: person=170cm, car=145cm, truck=200cm
- Calculates pixels-to-cm ratio from object bounding box
- Estimates water level at object's waterline
- Formula: `depth_cm = object_height_cm * (water_height_pixels / bbox_height_pixels)`

**Method 3: Hybrid Ensemble**
- Combines severity class, water percentage, object detections
- Weighted averaging:
  - If YOLO objects detected: 60% object-based, 40% severity-based
  - If no objects: 100% severity-based
  - Adjusts by water coverage percentage

**Output**:
```python
{
    'depth_cm': int,                  # Final depth estimate
    'depth_band': str,                # 'e.g., "20-50 cm"'
    'method': str,                    # Which method was used
    'details': {                      # Detailed breakdown
        'severity_depth': int,
        'object_based_depth': int,
        'water_percentage': float,
        'objects_used': [...]
    }
}
```

### **4. ObjectDetector** (`modules/object_detection.py`)
**Purpose**: Detect objects and provide anchor-based depth estimation

**YOLO Model**: YOLOv8 Nano (fastest variant)
- Pre-trained on COCO dataset
- 80+ object classes
- Real-time detection

**Detection Process**:
1. Load image and run through YOLOv8
2. Extract: class name, confidence, bounding box
3. Calculate bbox dimensions and center point
4. Attach known physical specs for detected class

**Known Object Specs**:
```python
{
    "person": {"height": 170, "width": 45},        # cm
    "car": {"height": 145, "width": 175},
    "truck": {"height": 200, "width": 250},
    "bus": {"height": 280, "width": 260},
    ...
}
```

**Depth Estimation from Objects**:
- **Waterline Detection**: Assume water touches bottom of object
- **Perspective Calculation**: Object size ratio → distance
- **Result**: Estimated water height in cm

**Output**:
```python
[
    {
        'class': 'person',
        'confidence': 0.83,
        'bbox': {
            'x1': 100, 'y1': 50,
            'x2': 200, 'y2': 300,
            'width': 100, 'height': 250,
            'center_x': 150, 'center_y': 175
        },
        'specs': {'height': 170, 'width': 45},
        'area_pixels': 25000
    },
    ...
]
```

### **5. FloodAnalyzer** (`modules/flood_analyzer.py`)
**Purpose**: Orchestrate the complete analysis pipeline

**Pipeline Logic**:
```python
analyze_bgr(image) → {
    1. Run water detection
    2. If no water: return early (skip expensive ML models)
    3. If water detected:
       a. Load severity predictor
       b. Get severity class & confidence
       c. Initialize hybrid depth estimator
       d. Estimate depth using multiple methods
       e. Return complete results
}
```

**Lazy Loading**: Models loaded only when needed
- Water detector: Always loaded
- Severity predictor: Loaded only if water detected
- Depth estimator: Loaded when hybrid estimation needed

**Efficiency**: Avoids loading expensive YOLO/ResNet when not needed

### **6. VideoFloodAnalyzer** (`modules/process_video.py`)
**Purpose**: Process video streams frame-by-frame

**Pipeline**:
1. Open video file with cv2.VideoCapture
2. For each frame (respecting skip_frames):
   - Run FloodAnalyzer on frame
   - Store results in DataFrame row
   - Optionally save annotated frame
3. Create output video from saved frames
4. Export DataFrame to CSV
5. Calculate statistics

### **7. S3Handler** (`modules/s3_handler.py`)
**Purpose**: AWS S3 integration for cloud storage

**Operations**:
- `read_image_from_s3()`: Download image, process, return
- `write_image_to_s3()`: Upload result image
- `read_video_from_s3()`: Download, process locally, cleanup
- `write_csv_to_s3()`: Upload analysis results
- Authentication via AWS credentials (env vars)

---

## 🎛️ COMMAND-LINE INTERFACE

```
Usage: python main.py <mode> <path> [options]

MODES:
  image  → Analyze single image
  video  → Process video file
  object → Detect objects with YOLO

SYNTAX:
  python main.py image <image_path> [--storage=local|aws]
  python main.py video <video_path> [output.csv] [skip_frames] [--storage=local|aws]
  python main.py object <image_path> [output.jpg] [--storage=local|aws]

EXAMPLES:
  Local mode (default):
    python main.py image test_images/flood.jpg
    python main.py video video.mp4
    python main.py object flood.jpg output.jpg

  AWS S3 mode:
    python main.py image images/flood.jpg --storage=aws
    python main.py video videos/flood.mp4 results.csv 2 --storage=aws
    python main.py object images/flood.jpg detected.jpg --storage=aws

STORAGE MODES:
  --storage=local  (DEFAULT)  → Use local files
  --storage=aws               → Use AWS S3 bucket
```

---

## 📊 DATA FLOW EXAMPLES

### Example 1: Image with Water and Person

```
INPUT: test_images/flood.jpg (512×384 pixels, contains person and water)
    ↓
WATER DETECTION:
  Method 1 (Color): ✓ Water detected
  Method 2 (Edge): ✓ Water detected
  Method 3 (Contrast): ✓ Water detected
  Method 4 (Line): ✗ No water line
  Method 5 (Depth): ✗ No depth map
  Method 6 (Ripple): ✓ Water ripples detected
  
  Votes: 4/6 methods agree → water_detected = TRUE
    ↓
SEVERITY CLASSIFICATION:
  ResNet18 output: [0.05, 0.15, 0.75, 0.04, 0.01]
  Argmax = 2 → "Moderate Flood", confidence = 75%
    ↓
DEPTH ESTIMATION (Hybrid):
  - Object Detection: Found 1 person
  - Person height: 170 cm, bbox height: 200 pixels
  - Ratio: 170/200 = 0.85 cm/pixel
  - Water level at: 180 pixels from top
  - Depth: 180 * 0.85 = 153 cm (OUTLIER)
  
  - Severity-based: Class 2 → 35 cm
  - Weighted: 60% object (153) + 40% severity (35) = 97 cm (adjusted)
  - Final: 35 cm (uses severity as it's more reliable)
    ↓
OUTPUT:
  Image: test_images/flood.jpg
  Water Detected: Yes
  Severity: 2 - Moderate Flood (75% confidence)
  Estimated Depth: 35 cm
  Flood Level: Moderate Flood
```

### Example 2: Video with 60 Frames (skip=2, process 30 frames)

```
INPUT: video.mp4 (1920×1080, 30 fps, 2 seconds = 60 frames)

Frame 0 (0.0s):  Water: NO   → Severity: N/A,   Depth: 0cm
Frame 2 (0.067s): Water: YES  → Severity: 2,    Depth: 35cm
Frame 4 (0.133s): Water: YES  → Severity: 2,    Depth: 35cm
Frame 6 (0.2s):   Water: YES  → Severity: 3,    Depth: 65cm
Frame 8 (0.267s): Water: YES  → Severity: 3,    Depth: 65cm
... (continue for all 60 frames)

STATISTICS:
  Total frames processed: 30
  Frames with water: 24
  Average water percentage: 65.3%
  
  Severity distribution:
    No Flood: 6 frames
    Moderate: 12 frames
    High: 18 frames
  
  Average depth (water frames): 53 cm

OUTPUT CSV: video_analysis.csv
  frame_number | time_seconds | water_detected | severity_class | depth_cm
  0            | 0.0          | False          | None           | 0
  2            | 0.067        | True           | 2              | 35
  4            | 0.133        | True           | 2              | 35
  6            | 0.2          | True           | 3              | 65
  ...

VISUALIZATION:
  output_frames/frame_0000.jpg (annotated)
  output_frames/frame_0005.jpg (annotated)
  ... (every 5 frames)
  output_frames/output_video.mp4 (all annotated frames stitched)
```

---

## 🚀 EXECUTION FLOW (Runtime)

```
START: main.py
    ↓
1. Parse command-line arguments
   - Extract mode: image/video/object
   - Extract path: file location
   - Extract storage: local or aws
    ↓
2. Initialize storage handler
   - If aws: S3Handler(credentials from env)
   - If local: Use native file I/O
    ↓
3. Route to appropriate processor
    │
    ├─ MODE = "image"
    │   └─ process_single_image()
    │       └─ FloodAnalyzer.analyze_bgr()
    │
    ├─ MODE = "video"
    │   └─ process_video_file()
    │       └─ VideoFloodAnalyzer.process_video()
    │
    └─ MODE = "object"
        └─ process_object_detection()
            └─ ObjectDetector.detect_objects()
    ↓
4. Display results to console
    ↓
5. Export/Upload (if needed)
    ↓
END: Print completion message
```

---

## 📈 PERFORMANCE CHARACTERISTICS

| Component | Time (ms) | GPU Speedup | Notes |
|-----------|-----------|------------|-------|
| Water Detection | 50-100 | 1x | Uses OpenCV only |
| Severity Classification | 200-500 | 2-3x | ResNet18 inference |
| YOLO Detection | 100-150 | 2-3x | YOLOv8 Nano |
| Depth Estimation | 10-20 | 1x | Mathematical |
| **Per Frame Total** | **400-800ms** | **2-3x** | Full pipeline |
| **Video (1 FPS)** | **1000ms** | **N/A** | Frame processing time |

---

## 💾 FILE STRUCTURE & OUTPUTS

```
wells_lab_aman/
├── main.py                           # Entry point
├── requirements.txt                  # Dependencies
├── severity_model.pth                # Trained ResNet18 weights
├── yolov8n.pt                        # YOLO model weights
├── test_images/                      # Test dataset
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ... (100+ images)
├── modules/
│   ├── __init__.py
│   ├── water_detection.py
│   ├── predict_image.py
│   ├── depth_band_estimator.py
│   ├── object_detection.py
│   ├── hybrid_depth_estimator.py
│   ├── flood_analyzer.py
│   ├── process_video.py
│   └── s3_handler.py
├── output_frames/                    # Generated video frames
│   ├── frame_0000.jpg
│   ├── frame_0005.jpg
│   └── output_video.mp4
├── video_analysis.csv                # Generated CSV results
├── objects_detected.jpg               # Generated detection image
└── fresh_env/                        # Virtual environment
```

---

## ✅ VERIFICATION CHECKLIST

- ✓ All imports successful
- ✓ Model files present (severity_model.pth, yolov8n.pt)
- ✓ Single image analysis working
- ✓ Object detection working
- ✓ Video processing pipeline ready
- ✓ AWS S3 integration available (optional)
- ✓ CSV export functionality ready
- ✓ Error handling and validation in place

---

## 📌 KEY DESIGN PATTERNS

1. **Lazy Loading**: Models loaded only when needed
2. **Ensemble Methods**: 6-method consensus for water detection
3. **Hybrid Estimation**: Multiple depth estimation methods combined
4. **Storage Abstraction**: Unified interface for local/cloud storage
5. **Error Resilience**: Fallback to severity-based depth if YOLO fails
6. **Early Exit**: Skip expensive ML if no water detected
7. **CSV Export**: Structured data export for analysis

---

## 🔧 TROUBLESHOOTING

| Issue | Cause | Solution |
|-------|-------|----------|
| "No module named cv2" | OpenCV not installed | `pip install opencv-python` |
| "Model file not found" | Missing severity_model.pth | Place in project root |
| "CUDA out of memory" | Large image/batch size | Process fewer frames (skip_frames=2+) |
| "S3 connection failed" | Missing AWS credentials | Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY |
| "YOLO model download fails" | Network issue | Manual download to ~/.yolov8/ |
| Slow performance | CPU-only mode | Install CUDA toolkit |

---

## 📝 CONCLUSION

This system provides an **end-to-end solution** for flood detection and depth estimation:

1. **Water Detection**: Fast ensemble method with 6 techniques
2. **Severity Classification**: Deep learning with ResNet18
3. **Depth Estimation**: Hybrid approach combining physics and ML
4. **Object Detection**: YOLO for context and anchor-based depth
5. **Cloud Integration**: AWS S3 for scalable deployment
6. **Batch Processing**: Video and multi-image support

The modular design allows easy extension and adaptation to specific use cases.


