# ENHANCED FLOOD DEPTH ESTIMATION SYSTEM
## Multi-Image Temporal Analysis with Hallucination Prevention

---

## 🎯 OVERVIEW

Your flood depth estimation system has been significantly enhanced to address the requirements:

✅ **Time-Interval Analysis**: Collects 5+ images from the same camera within 5-15 minute windows  
✅ **Multi-Reference Object Validation**: Uses person, car, bus, motorcycle, and walls as calibration points  
✅ **Hallucination Prevention**: Requires multiple anchors + water consensus before confirming flooding  
✅ **Camera & Location Tracking**: Each image is tagged with camera_id and location metadata  
✅ **Temporal Consensus**: Aggregates depth estimates from multiple reference objects  

---

## 🏗️ ARCHITECTURE CHANGES

### 1. **New Database Models**

#### `CameraLocation`
Tracks unique camera deployment sites:
```python
CameraLocation
  - camera_id: str (unique)
  - location_name: str
  - latitude/longitude: float (optional GPS)
  - created_at: timestamp
```

#### `FloodInundationTelemetry` (Enhanced)
Added fields for multi-anchor validation:
```python
FloodInundationTelemetry
  - camera: FK → CameraLocation
  - detected_reference_objects: list of str  # ["person", "car"]
  - num_reference_objects: int
  - is_water_confirmed: bool  # ← HALLUCINATION PREVENTION
```

#### `TemporalFloodSequence` (New)
Groups images from same camera over 5-15 min intervals:
```python
TemporalFloodSequence
  - camera: FK → CameraLocation
  - sequence_start/end: timestamp
  - image_count: int
  - average_depth_cm: float  # Consensus from multiple anchors
  - detected_anchor_types: list  # ["person", "car", "bus"]
  - consensus_water_present: bool  # VALIDATED
  - confidence_score: float (0.0-1.0)
```

---

## 📊 HOW IT WORKS

### **Single Image Processing** (Stage 1)

```python
process_and_refine_telemetry(image_filepath, filename, camera_id="camera_01")
```

Each image is analyzed for:
1. **Flood detection probability** (0.0-1.0)
2. **Depth estimation** using monocular depth + YOLO object detection
3. **Reference objects detected** (person, car, bus, motorcycle, truck)
4. **Single-image water confirmation** (requires 2+ anchors OR high confidence + water detection)

**Output:**
```json
{
  "status": "success",
  "record_id": 123,
  "depth_cm": 45.2,
  "reference_objects": ["car", "person"],
  "is_water_confirmed": true
}
```

---

### **Temporal Sequence Analysis** (Stage 2)

```python
analyze_temporal_sequence(camera_id="camera_01", time_window_minutes=15)
```

When 5+ images arrive from same camera within 5-15 minutes:

1. **Fetch recent images** from that camera
2. **Multi-Anchor Validation**:
   - Count unique reference object types (person, car, bus, etc.)
   - Check water consensus across all images
   - Validate depth consistency
3. **Confidence Scoring**:
   - 1 anchor type + 5+ images = LOW confidence
   - 2 anchor types + 3+ images = MEDIUM confidence  
   - 3+ anchor types = HIGH confidence
4. **Calculate Consensus Depth**:
   - Average depth measurements across all reference objects
   - Weighted by confidence scores
   - Detect outliers

**Hallucination Prevention:**
```
❌ REJECTED: No water detected (water_consensus_pct < 40%)
❌ REJECTED: Only 1 reference object + < 5 images
✅ ACCEPTED: 2 anchor types + 3+ images + water consensus
✅ ACCEPTED: 3+ anchor types (any number of images)
```

---

## 📡 API ENDPOINTS

### **1. Single Image Upload** (Existing)
```bash
POST /api/v1/estimate/
Content-Type: multipart/form-data

image: <file>
camera_id: "intersection_01"  # NEW: Track which camera
location_name: "Main Street & 5th Ave"  # Optional
latitude: 40.7128
longitude: -74.0060
context: "Heavy rainfall reported"
```

**Response:**
```json
{
  "status": "buffered",
  "message": "Frame buffered. Queue depth: 3/5",
  "camera_id": "intersection_01",
  "queue_percentage": 60.0
}
```

When queue reaches 5 frames:
```json
{
  "status": "processing",
  "message": "Buffer full (5 frames). Triggering batch inference + temporal analysis.",
  "camera_id": "intersection_01"
}
```

---

### **2. Get Temporal Sequence** (New)
```bash
GET /api/v1/temporal/intersection_01/
?time_window=15
```

**Response:**
```json
{
  "status": "success",
  "sequence_id": 42,
  "camera_id": "intersection_01",
  "num_images": 5,
  "time_span_minutes": 12.3,
  "average_depth_cm": 42.5,
  "max_depth_cm": 55.2,
  "min_depth_cm": 35.1,
  "detected_anchor_types": ["car", "person", "bus"],
  "consensus_water_present": true,
  "confidence_score": 0.856,
  "sequence_start": "2024-06-04T10:30:00Z",
  "sequence_end": "2024-06-04T10:42:20Z"
}
```

---

### **3. Trigger Temporal Analysis** (New)
```bash
POST /api/v1/temporal/intersection_01/analyze/
time_window=15
```

Manually queue analysis (useful for testing).

---

### **4. Get Camera Statistics** (New)
```bash
GET /api/v1/camera/intersection_01/stats/
?hours=24
```

**Response:**
```json
{
  "status": "success",
  "camera_id": "intersection_01",
  "camera_name": "Main Street & 5th Ave",
  "hours_analyzed": 24,
  "total_images": 47,
  "water_confirmed_images": 18,
  "avg_depth_cm": 38.5,
  "max_depth_cm": 62.3,
  "temporal_sequences": 6
}
```

---

## 💡 EXAMPLE WORKFLOW: 10 Images from Same Location

### **Scenario:**
You send 10 images from camera `intersection_01` (same location) over 12 minutes.
Each image is 2 minutes apart.

### **What Happens:**

**Images 1-5** (0-8 minutes):
- Buffered in Redis
- After image 5, temporal analysis triggers

**Temporal Analysis Results:**
```python
# Frame-by-frame analysis:
Image 1: water_prob=45%, depth=35cm, objects=["car"]
Image 2: water_prob=52%, depth=40cm, objects=["car", "person"]  
Image 3: water_prob=48%, depth=38cm, objects=["car", "bus"]
Image 4: water_prob=55%, depth=42cm, objects=["car", "person"]
Image 5: water_prob=50%, depth=39cm, objects=["car"]

# Temporal Consensus:
✅ Water consensus: 50% average (> 40% threshold) → CONFIRMED
✅ Reference objects: 3 unique types (car, person, bus) → HIGH CONFIDENCE
✅ Depth consistency: std_dev = 1.8cm (low variance) → RELIABLE
✅ Result: VALIDATED water detection

Final Output:
{
  "sequence_id": 15,
  "num_images": 5,
  "consensus_depth_cm": 38.8,  # Average across anchors
  "confidence_level": "high",
  "consensus_water_present": true,
  "detected_anchor_types": ["car", "person", "bus"],
  "final_risk_assessment": {
    "level": "MODERATE",
    "reason": "Depth 38.8cm - small vehicles compromised"
  }
}
```

**Images 6-10** (9-12 minutes):
- Buffered separately
- When 5th image arrives (image 10), triggers another temporal analysis
- Can be compared with previous sequence to detect depth changes

---

## 🚨 HALLUCINATION PREVENTION LOGIC

### **Case 1: Single Car Detected**
```
Image detected: car (1 reference object)
Water probability: 35% (LOW)
Depth estimate: 45cm

❌ HALLUCINATION DETECTED
Reason: Only 1 reference object + water confidence below 40%
Action: Mark as LOW confidence, require 5+ images for validation
```

### **Case 2: No Water Detected**
```
Images: 5 frames, 3 reference objects (car, person, bus)
Average water probability: 25% (VERY LOW)

❌ HALLUCINATION PREVENTED
Reason: No consensus on water presence (25% < 40% threshold)
Action: Do NOT mark is_water_confirmed = true
Risk: "UNVERIFIED - Insufficient data to confirm water"
```

### **Case 3: Clear Flooding (NO Hallucination)**
```
Images: 10 frames, 4 reference objects
Water consensus: 72% (all high confidence)
Depth estimates: 55, 58, 54, 56, 57cm (std_dev = 1.2)

✅ VALIDATED
Consensus depth: 56cm
Confidence: HIGH (0.92)
Risk: "CRITICAL - Depth 56cm, most vehicles risk stalling"
```

---

## 🔧 TECHNICAL DETAILS

### **Multi-Anchor Depth Calculation**

For each reference object type (person, car, bus, etc.):
```python
weighted_depth = Σ(depth_i × confidence_i) / Σ(confidence_i)
```

Example with 5 images:
```
Person measurements: [50, 48, 49cm] → mean: 49cm
Car measurements: [45, 46, 47cm] → mean: 46cm
Bus measurement: [44cm] → mean: 44cm

Consensus depth = (49 + 46 + 44) / 3 = 46.3cm
```

### **Confidence Score Components**

```
confidence = (0.4 × anchor_factor) + 
             (0.3 × water_consensus) + 
             (0.3 × image_count_factor)

Where:
  - anchor_factor = min(num_unique_anchors / 3, 1.0)
  - water_consensus = average_water_probability
  - image_count_factor = min(num_images / 10, 1.0)
```

---

## 🚀 DEPLOYMENT STEPS

### **1. Run Migrations**
```bash
python manage.py migrate
```

This creates:
- `CameraLocation` table
- Enhanced `FloodInundationTelemetry` fields
- `TemporalFloodSequence` table

### **2. (Optional) Create Initial Camera Locations**
```python
from flood_api.models import CameraLocation

CameraLocation.objects.create(
    camera_id="intersection_01",
    location_name="Main Street & 5th Ave",
    latitude=40.7128,
    longitude=-74.0060,
    description="High-priority intersection"
)
```

### **3. Test the System**

Upload images via API:
```bash
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/v1/estimate/ \
    -F "image=@test_image_$i.jpg" \
    -F "camera_id=test_camera_01" \
    -F "location_name=Test Location"
  sleep 120  # 2 minute interval
done
```

### **4. Check Results**
```bash
# Get latest temporal sequence
curl http://localhost:8000/api/v1/temporal/test_camera_01/

# Get camera statistics
curl http://localhost:8000/api/v1/camera/test_camera_01/stats/?hours=1
```

---

## 📝 DATABASE QUERIES

### **Find All Water-Confirmed Flooding**
```python
from flood_api.models import FloodInundationTelemetry

validated_records = FloodInundationTelemetry.objects.filter(
    is_water_confirmed=True,
    computed_depth_cm__gte=30
).order_by('-timestamp')
```

### **Get High-Confidence Temporal Sequences**
```python
from flood_api.models import TemporalFloodSequence

high_confidence_sequences = TemporalFloodSequence.objects.filter(
    confidence_score__gte=0.8,
    consensus_water_present=True
).select_related('camera').order_by('-sequence_start')
```

### **Find Cameras with Recent Flooding**
```python
from django.utils import timezone
from datetime import timedelta

recent_time = timezone.now() - timedelta(hours=1)
flooding_cameras = CameraLocation.objects.filter(
    floodinundationtelemetry__is_water_confirmed=True,
    floodinundationtelemetry__timestamp__gte=recent_time
).distinct()
```

---

## 🎯 KEY IMPROVEMENTS

| Feature | Before | After |
|---------|--------|-------|
| Multi-image analysis | ❌ No | ✅ Yes (5-15 min windows) |
| Reference object validation | ❌ Single object | ✅ Multiple anchors required |
| Hallucination prevention | ❌ None | ✅ Multi-anchor + consensus |
| Camera tracking | ❌ No | ✅ Yes, with GPS support |
| Depth consensus | ❌ Single estimate | ✅ Weighted average by object |
| Time-interval batching | ❌ Manual | ✅ Automatic 5-frame triggers |
| Risk assessment | ❌ Basic | ✅ Confidence-weighted |

---

## ⚠️ IMPORTANT NOTES

1. **PostgreSQL ArrayField**: The new models use Django's `ArrayField` for storing object lists. Ensure your database is PostgreSQL (or add a fallback for SQLite).

2. **Redis Buffering**: Images are buffered in Redis for 15 minutes. Ensure Redis is running and accessible.

3. **Celery Tasks**: Both `process_and_refine_telemetry` and `analyze_temporal_sequence` are async. Monitor Celery workers during testing.

4. **Migration**: After deploying, run `python manage.py migrate` to create new tables and add new fields.

5. **Test Data**: Use the included test scripts to generate sample images if needed.

---

## 📞 TROUBLESHOOTING

### **Temporal analysis not triggering?**
- Check Redis connection: `redis-cli ping`
- Verify Celery worker is running
- Check logs for `analyze_temporal_sequence` task

### **Camera not found when submitting images?**
- This is OK! The system auto-creates camera locations on first submission
- Verify with: `curl http://localhost:8000/api/v1/camera/YOUR_CAMERA_ID/stats/`

### **Depth estimates seem off?**
- Ensure reference objects (person, car, bus) are clearly visible in images
- Check YOLO detection: images should have unobstructed objects
- Use manual temporal analysis to review: `POST /api/v1/temporal/camera_id/analyze/`

---

## 📚 REFERENCES

- **Depth Estimation**: Depth Anything V2 (https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2)
- **Object Detection**: YOLOv8 (https://github.com/ultralytics/ultralytics)
- **Water Detection**: Custom MobileNetV3 classifier (your `lightweight_flood_classifier.pt`)
