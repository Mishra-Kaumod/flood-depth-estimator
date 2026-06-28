import cv2
import os
import torch
from celery import shared_task
from transformers import pipeline
from core_logic import TripleEnginePipeline, estimate_flood_depth
from .models import FloodInundationTelemetry, CameraLocation
from .temporal_analysis import TemporalFloodAnalyzer
from .services.prediction_policy import harmonize_prediction

print("[+] Initializing Core Vision Models...")
ml_pipeline = TripleEnginePipeline()
temporal_analyzer = TemporalFloodAnalyzer()

print("[+] Initializing Local Open-Weight LLM (Zero Marginal Cost)...")
# Loading a highly optimized 1.5B parameter model into memory.
if os.getenv("ENABLE_LOCAL_LLM", "0").lower() in ("1", "true", "yes"):
    try:
        if torch.cuda.is_available():
            llm_generator = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            llm_generator = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                device=-1
            )
    except Exception as e:
        print(f"[!] LLM failed to load. Defaulting to strict math thresholds. Error: {e}")
        llm_generator = None
else:
    print("[!] Local LLM disabled by default. Using strict math thresholds only.")
    llm_generator = None

@shared_task(bind=True, max_retries=3)
def process_and_refine_telemetry(self, image_filepath, filename, external_context="", camera_id=None):
    """
    ENHANCED: Executes Vision Math -> Multi-Anchor Validation -> LLM Refinement
    
    New Parameters:
        camera_id: Identifies which camera took the image (for temporal grouping)
    """
    # --- STAGE 0: GET OR CREATE CAMERA LOCATION ---
    camera = None
    if camera_id:
        try:
            camera = CameraLocation.objects.get(camera_id=camera_id)
        except CameraLocation.DoesNotExist:
            # Auto-create camera location if not exists
            camera = CameraLocation.objects.create(
                camera_id=camera_id,
                location_name=f"Location {camera_id}",
                description=f"Auto-created from upload"
            )
    
    # --- STAGE 1: COMPUTER VISION MATH ---
    img_matrix = cv2.imread(image_filepath)
    if img_matrix is None:
        return {"status": "error", "message": "Corrupted image file"}

    # Get flood classification confidence
    flood_prob = ml_pipeline.predict_flood_probability(img_matrix)
    
    # Use in-repo ensemble depth engine for depth + strategy
    cv_results = estimate_flood_depth(img_matrix)
    if cv_results.get("status") != "success":
        return {"status": "error", "message": "Depth estimation failed"}

    raw_depth = float(cv_results.get("estimated_depth_cm", 0.0))
    ensemble_confidence = float(cv_results.get("ensemble_confidence", 0.0))
    raw_confidence = max(float(flood_prob), ensemble_confidence)
    strategy = cv_results.get("calculation_mode", "Ensemble")

    if "No anchor" in strategy:
        detected_anchors = []
    else:
        detected_anchors = ["anchor_detected"]
    num_anchors = len(detected_anchors)
    is_fallback = num_anchors == 0

    # --- STAGE 2: CONSISTENCY POLICY ---
    policy = harmonize_prediction(
        raw_depth_cm=raw_depth,
        water_pct=flood_prob * 100.0,
        raw_confidence=raw_confidence,
        num_anchors=num_anchors,
    )
    raw_depth = policy["depth_cm"]
    is_water_confirmed = policy["is_water_confirmed"]
    hallucination_warning = policy["warning"]
    
    # --- STAGE 3: HEURISTIC GATING & LLM REFINEMENT ---
    if not is_water_confirmed:
        refined_risk = "Low"
        llm_justification = "System: Depth is structurally safe for standard transit."
    elif raw_depth <= 20.0:
        refined_risk = "Low"
        llm_justification = "System: Depth is structurally safe for standard transit."
    
    elif raw_depth >= 60.0:
        refined_risk = "Critical"
        llm_justification = "System: Depth exceeds critical safety limits. Immediate closure advised."
        
    else:
        # THE GRAY ZONE (21cm - 59cm): Trigger the Local LLM for contextual nuance
        if llm_generator:
            prompt = f"Flood depth is {raw_depth}cm. Ref objects: {num_anchors}. Context: '{external_context}'. Risk level? Response format: [Low/Moderate/Critical]|[1-sentence reason]"
            
            messages = [{"role": "user", "content": prompt}]
            output = llm_generator(messages, max_new_tokens=50, temperature=0.1)
            raw_text = output[0]['generated_text'][-1]['content']
            
            try:
                parts = raw_text.split('|')
                refined_risk = parts[0].strip()
                llm_justification = parts[1].strip()
            except:
                refined_risk = "Moderate"
                llm_justification = "LLM parsing error. Math indicates hazard."
        else:
            refined_risk = "Moderate"
            llm_justification = "LLM offline. Manual assessment needed."

    # --- STAGE 4: ENTERPRISE DATABASE PERSISTENCE ---
    record = FloodInundationTelemetry.objects.create(
        image_name=filename,
        camera=camera,
        strategy_applied=strategy,
        surface_water_confirmed_pct=round(flood_prob * 100, 2),
        computed_depth_cm=raw_depth,
        system_confidence_score_pct=round(raw_confidence * 100, 2),
        detected_reference_objects=detected_anchors,
        num_reference_objects=num_anchors,
        is_water_confirmed=is_water_confirmed,
        safety_risk_assessment=f"{refined_risk} - {llm_justification}{' ' + hallucination_warning if hallucination_warning else ''}"
    )

    # Cleanup temporary file
    if os.path.exists(image_filepath):
        os.remove(image_filepath)

    return {
        "status": "success",
        "record_id": record.id,
        "depth_cm": raw_depth,
        "reference_objects": detected_anchors,
        "is_water_confirmed": is_water_confirmed
    }


@shared_task(bind=True, max_retries=3)
def analyze_temporal_sequence(self, camera_id, time_window_minutes=15):
    """
    ENHANCED: Analyzes a temporal sequence from a camera with multi-anchor validation.
    
    This task:
    1. Fetches all images from the camera in the last N minutes
    2. Validates water presence using multiple reference objects
    3. Calculates consensus depth from multiple anchor types
    4. Prevents hallucination by requiring multiple objects/images
    
    Args:
        camera_id: Camera identifier (e.g., "intersection_01")
        time_window_minutes: Time window for analysis (default 5-15 minutes)
    
    Returns:
        dict with temporal sequence analysis results
    """
    result = temporal_analyzer.create_temporal_sequence(
        camera_id=camera_id,
        time_window_minutes=time_window_minutes
    )
    
    if result.get('status') == 'error':
        return {"status": "error", "message": result.get('message')}
    
    if result.get('status') == 'insufficient_data':
        return {"status": "insufficient_data", "message": result.get('message')}
    
    # Log the sequence analysis
    print(f"\n📊 TEMPORAL SEQUENCE ANALYSIS - {camera_id}")
    print(f"   Images: {result['num_images']} over {result['time_span_minutes']} minutes")
    print(f"   Reference Objects: {result['validation']['num_unique_anchors']} types")
    print(f"   Water Consensus: {result['validation']['water_consensus_pct']}%")
    print(f"   Validation: {result['validation']['confidence_level']}")
    if result['consensus_depth_cm']:
        print(f"   Consensus Depth: {result['consensus_depth_cm']}cm")
    print(f"   Risk Level: {result['final_risk_assessment']['level']}")
    print()
    
    return result
