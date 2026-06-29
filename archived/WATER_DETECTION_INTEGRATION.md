# WATER DETECTION INTEGRATION GUIDE
## How to Add Advanced Detection to Your API

---

## 🎯 QUICK INTEGRATION (5 minutes)

### Step 1: Update Your Celery Task

In `flood_api/tasks.py`, replace the water detection section:

```python
# OLD (single method)
flood_prob = ml_pipeline.predict_flood_probability(img_matrix)

# NEW (multi-method)
from test_water_detection_simple import (
    simple_water_detection,
    detect_horizontal_edges
)

flood_prob = ml_pipeline.predict_flood_probability(img_matrix)
color_result = simple_water_detection(img_matrix)
edge_result = detect_horizontal_edges(img_matrix)

# Voting
water_votes = 0
if flood_prob > 0.4:
    water_votes += 1
if color_result['water_detected']:
    water_votes += 1
if edge_result['water_detected']:
    water_votes += 1

is_water_confirmed = water_votes >= 2
water_confidence = water_votes / 3
```

### Step 2: Update Database Record

```python
record = FloodInundationTelemetry.objects.create(
    # ... existing fields ...
    surface_water_confirmed_pct=round(water_confidence * 100, 2),
    computed_depth_cm=raw_depth,
    system_confidence_score_pct=round(raw_confidence * 100, 2),
    detected_reference_objects=detected_anchors,
    num_reference_objects=num_anchors,
    is_water_confirmed=is_water_confirmed,  # ← NEW: only True if consensus
    safety_risk_assessment=f"{refined_risk} - {confidence_message}"
)
```

### Step 3: Update API Response

```python
@csrf_exempt
def high_speed_api_endpoint(request):
    # ... existing code ...
    
    return JsonResponse({
        "status": "success",
        "depth_cm": raw_depth,
        "water_detected": is_water_confirmed,      # ← NEW
        "water_confidence": water_confidence,       # ← NEW
        "method_agreement": f"{water_votes}/3",    # ← NEW
        "risk_level": refined_risk
    })
```

---

## 📊 FULL INTEGRATION EXAMPLE

### New Enhanced Task

```python
# In flood_api/tasks.py

import cv2
import os
import torch
from celery import shared_task
from transformers import pipeline
from django.utils import timezone

from core_logic import TripleEnginePipeline, estimate_flood_depth
from cv_engine import FloodDepthEngine
from test_water_detection_simple import (
    simple_water_detection,
    detect_horizontal_edges
)
from .models import FloodInundationTelemetry, CameraLocation
from .temporal_analysis import TemporalFloodAnalyzer

ml_pipeline = TripleEnginePipeline()
cv_engine = FloodDepthEngine()
temporal_analyzer = TemporalFloodAnalyzer()

@shared_task(bind=True, max_retries=3)
def process_and_refine_telemetry(
    self, 
    image_filepath, 
    filename, 
    external_context="", 
    camera_id=None
):
    """
    ENHANCED: Multi-method water detection with consensus voting.
    """
    # Get or create camera
    camera = None
    if camera_id:
        camera, _ = CameraLocation.objects.get_or_create(
            camera_id=camera_id,
            defaults={'location_name': f'Location {camera_id}'}
        )
    
    # Load image
    img_matrix = cv2.imread(image_filepath)
    if img_matrix is None:
        return {"status": "error", "message": "Corrupted image"}
    
    # --- STAGE 1: DEPTH ESTIMATION ---
    flood_prob = ml_pipeline.predict_flood_probability(img_matrix)
    cv_results = cv_engine.process_frame(img_matrix)
    raw_depth = cv_results["calculated_depth_cm"]
    detected_anchors = cv_results["anchors_tracked"]
    
    # --- STAGE 2: MULTI-METHOD WATER DETECTION ---
    # Method 1: Neural classifier
    classifier_vote = flood_prob > 0.4
    
    # Method 2: Color analysis
    color_result = simple_water_detection(img_matrix)
    color_vote = color_result['water_detected']
    
    # Method 3: Edge detection
    edge_result = detect_horizontal_edges(img_matrix)
    edge_vote = edge_result['water_detected']
    
    # Consensus voting
    water_votes = sum([classifier_vote, color_vote, edge_vote])
    is_water_confirmed = water_votes >= 2
    water_confidence = water_votes / 3
    
    print(f"[Water Detection Votes] Classifier: {classifier_vote} | "
          f"Color: {color_vote} | Edge: {edge_vote} → "
          f"Consensus: {is_water_confirmed}")
    
    # --- STAGE 3: RISK ASSESSMENT ---
    if is_water_confirmed:
        if raw_depth > 60:
            risk_level = "CRITICAL"
            message = "Immediate closure recommended"
        elif raw_depth > 30:
            risk_level = "HIGH"
            message = "Most vehicles at risk"
        else:
            risk_level = "MODERATE"
            message = "Small vehicles compromised"
    else:
        risk_level = "LOW"
        message = "No flooding detected"
    
    # --- STAGE 4: DATABASE PERSISTENCE ---
    record = FloodInundationTelemetry.objects.create(
        image_name=filename,
        camera=camera,
        strategy_applied=cv_results['strategy_applied'],
        surface_water_confirmed_pct=round(flood_prob * 100, 2),
        computed_depth_cm=raw_depth,
        system_confidence_score_pct=round(water_confidence * 100, 2),
        detected_reference_objects=detected_anchors,
        num_reference_objects=len(set(detected_anchors)),
        is_water_confirmed=is_water_confirmed,
        safety_risk_assessment=f"{risk_level} - {message}"
    )
    
    # Cleanup
    if os.path.exists(image_filepath):
        os.remove(image_filepath)
    
    return {
        "status": "success",
        "record_id": record.id,
        "water_detected": is_water_confirmed,
        "water_confidence": round(water_confidence, 2),
        "depth_cm": raw_depth,
        "risk_level": risk_level,
        "method_votes": {
            "classifier": classifier_vote,
            "color": color_vote,
            "edges": edge_vote,
            "consensus": is_water_confirmed
        }
    }
```

### Updated API Endpoint

```python
# In flood_api/views.py

@csrf_exempt
def high_speed_api_endpoint(request):
    """Enhanced with water detection results."""
    if request.method != "POST":
        return JsonResponse({"status": "failed"}, status=405)
    
    try:
        uploaded_file = request.FILES.get("image")
        camera_id = request.POST.get("camera_id", "default")
        
        if not uploaded_file:
            return JsonResponse(
                {"status": "failed", "error": "No image"}, 
                status=400
            )
        
        # Save and queue
        temp_path = f"/tmp/{uuid.uuid4()}_{uploaded_file.name}"
        with open(temp_path, 'wb+') as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)
        
        # Queue task
        task_result = process_and_refine_telemetry.delay(
            image_filepath=temp_path,
            filename=uploaded_file.name,
            external_context=request.POST.get("context", ""),
            camera_id=camera_id
        )
        
        return JsonResponse({
            "status": "processing",
            "message": "Image queued for analysis",
            "task_id": task_result.id,
            "camera_id": camera_id
        }, status=202)
    
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt  
def get_analysis_result(request, task_id):
    """Retrieve results of image analysis."""
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id)
    
    if task.state == 'PENDING':
        return JsonResponse({
            "status": "processing",
            "task_id": task_id
        }, status=202)
    
    elif task.state == 'SUCCESS':
        result = task.result
        return JsonResponse({
            "status": "success",
            "result": result,
            "water_detected": result.get('water_detected'),
            "water_confidence": result.get('water_confidence'),
            "depth_cm": result.get('depth_cm'),
            "risk_level": result.get('risk_level'),
            "method_votes": result.get('method_votes')
        }, status=200)
    
    else:
        return JsonResponse({
            "status": "failed",
            "error": task.info
        }, status=500)
```

---

## 🔌 API USAGE EXAMPLES

### Upload Image for Analysis

```bash
curl -X POST http://localhost:8000/api/v1/estimate/ \
  -F "image=@flood.jpg" \
  -F "camera_id=intersection_01" \
  -F "context=Heavy rainfall reported"
```

**Response:**
```json
{
  "status": "processing",
  "task_id": "a1b2c3d4-e5f6...",
  "message": "Image queued for analysis",
  "camera_id": "intersection_01"
}
```

### Get Results

```bash
curl http://localhost:8000/api/v1/result/a1b2c3d4-e5f6.../
```

**Response:**
```json
{
  "status": "success",
  "water_detected": true,
  "water_confidence": 0.67,
  "depth_cm": 45.3,
  "risk_level": "HIGH",
  "method_votes": {
    "classifier": true,
    "color": true,
    "edges": false,
    "consensus": true
  }
}
```

---

## 📊 DATABASE FIELDS TO ADD

If you haven't migrated yet, add these to your model:

```python
class FloodInundationTelemetry(models.Model):
    # ... existing fields ...
    
    # NEW FIELDS:
    detected_reference_objects = ArrayField(
        models.CharField(max_length=50),
        default=list,
        blank=True,
        help_text="Objects detected (person, car, bus, etc.)"
    )
    num_reference_objects = models.IntegerField(
        default=0,
        help_text="Count of unique reference object types"
    )
    is_water_confirmed = models.BooleanField(
        default=False,
        help_text="True only if multi-method consensus confirms water"
    )
```

---

## 🧪 TESTING YOUR INTEGRATION

### Test 1: Known Water Image

```python
# Run your test
python test_water_detection_simple.py

# Expect: 3/3 methods detect water
# Expect: is_water_confirmed = True
# Expect: Depth estimate is reasonable
```

### Test 2: Known No-Water Image

```python
# Create test image with no water
# Run: python test_water_detection_simple.py

# Expect: <2/3 methods detect water
# Expect: is_water_confirmed = False
# Expect: "No flooding" in result
```

### Test 3: Edge Cases

```python
# Test scenarios:
# - Rainy day (wet pavement, no flood)
# - Night time (dark conditions)
# - Heavy reflection (glass, metal)
# - Partial visibility (camera partially blocked)
```

---

## 🚀 DEPLOYMENT CHECKLIST

- [ ] Copy `test_water_detection_simple.py` functions into `tasks.py`
- [ ] Update database model with new fields
- [ ] Create and run migrations
- [ ] Update `process_and_refine_telemetry` task
- [ ] Update API endpoint response format
- [ ] Test with known water images
- [ ] Test with known no-water images
- [ ] Deploy to staging environment
- [ ] Monitor logs for any issues
- [ ] Deploy to production

---

## 📈 PERFORMANCE NOTES

| Method | Time | Memory |
|--------|------|--------|
| Classifier | 10-50ms | 50MB |
| Depth Engine | 100-200ms | 200MB |
| Color Analysis | 5-10ms | 10MB |
| Edge Detection | 10-20ms | 10MB |
| **Total** | **150-300ms** | **270MB** |

For batch processing (5+ images):
- Reuse models (don't reload per image)
- Process in parallel if possible
- Cache depth maps for multiple object detections

---

## 🐛 TROUBLESHOOTING

**Q: System always says "no water"**
A: Lower threshold from `>= 2` votes to `>= 1` temporarily, check which method is failing

**Q: Memory usage too high**
A: Process images in smaller batches, unload models between uses

**Q: Slow inference**
A: Use model quantization (INT8), reduce image resolution

**Q: False positives on rainy days**
A: Requires all 3 methods to agree (raise threshold to `>= 3`)

---

## 📚 REFERENCE

- Full water detection module: `water_detection.py` (6 methods)
- Simple integration: `test_water_detection_simple.py` (4 methods)
- Temporal analysis: `temporal_analysis.py` (for time-series)
- Guide: `WATER_DETECTION_GUIDE.md` (detailed explanation)

---

**Ready to deploy!** ✅
