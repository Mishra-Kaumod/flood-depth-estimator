# flood_api/views.py
import os
import uuid
import json
import redis
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg, Max
from django.utils import timezone
from datetime import timedelta
from .tasks import process_and_refine_telemetry, analyze_temporal_sequence
from .models import FloodInundationTelemetry, CameraLocation, TemporalFloodSequence

# Connect to the Redis instance
redis_client = redis.StrictRedis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))

def dashboard_view(request):
    """
    Renders the web interface. Manual uploads here bypass the 
    batch-buffer and go straight to the ML/LLM worker.
    """
    context = {}
    if request.method == "POST" and request.FILES.get("image_file"):
        try:
            uploaded_file = request.FILES["image_file"]
            camera_id = request.POST.get("camera_id", "manual_upload")
            
            # Save file temporarily for the background worker
            temp_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
            temp_path = os.path.join(settings.BASE_DIR, 'tmp', temp_filename)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            
            # Dispatch directly to the background Celery worker with camera_id
            process_and_refine_telemetry.delay(
                image_filepath=temp_path,
                filename=uploaded_file.name,
                external_context="Manual Web Dashboard Upload",
                camera_id=camera_id
            )
            
            context["success"] = True
            context["message"] = "Payload accepted! The AI workers are processing it in the background. Refresh the page in a few seconds to see the results in the log below."
        except Exception as e:
            context["error"] = f"Failed to process file: {str(e)}"
            
    # Pull the latest telemetry from the database for the UI table
    context["historical_records"] = FloodInundationTelemetry.objects.all()[:20]
    context["cameras"] = CameraLocation.objects.all()
    
    return render(request, "flood_api/dashboard.html", context)

@csrf_exempt
def high_speed_api_endpoint(request):
    """
    ENHANCED: High-volume endpoint for municipal cameras. Uses temporal batching 
    to prevent queue flooding during a storm.
    
    POST Parameters:
        - image: Image file
        - camera_id: Camera identifier (e.g., "intersection_01")
        - location_name: Human-readable location (optional)
        - latitude: GPS latitude (optional)
        - longitude: GPS longitude (optional)
        - context: Additional context about the flood (optional)
    """
    if request.method != "POST":
        return JsonResponse({"status": "failed"}, status=405)
        
    try:
        uploaded_file = request.FILES.get("image")
        camera_id = request.POST.get("camera_id", "intersection_01")
        location_name = request.POST.get("location_name", f"Location {camera_id}")
        latitude = request.POST.get("latitude", None)
        longitude = request.POST.get("longitude", None)
        external_context = request.POST.get("context", "")
        
        if not uploaded_file:
            return JsonResponse({"status": "failed", "error": "No image payload"}, status=400)

        # Create or update camera location
        camera, created = CameraLocation.objects.get_or_create(
            camera_id=camera_id,
            defaults={
                'location_name': location_name,
                'latitude': float(latitude) if latitude else None,
                'longitude': float(longitude) if longitude else None,
            }
        )
        
        temp_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        temp_path = os.path.join(settings.BASE_DIR, 'tmp', temp_filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        payload = {
            "image_filepath": temp_path,
            "filename": uploaded_file.name,
            "external_context": external_context,
            "camera_id": camera_id
        }
        
        redis_list_key = f"camera_buffer:{camera_id}"
        redis_client.lpush(redis_list_key, json.dumps(payload))
        
        # Set expiration so old frames don't linger
        redis_client.expire(redis_list_key, 900)  # 15 minutes
        
        current_queue_depth = redis_client.llen(redis_list_key)
        
        # Buffer Trigger: Run inference if we hit 5 frames (5-15 min interval)
        if current_queue_depth >= 5:
            latest_payload = json.loads(redis_client.lpop(redis_list_key))
            
            process_and_refine_telemetry.delay(
                image_filepath=latest_payload["image_filepath"],
                filename=latest_payload["filename"],
                external_context=latest_payload["external_context"],
                camera_id=latest_payload["camera_id"]
            )
            
            # Trigger temporal analysis on sequence
            analyze_temporal_sequence.delay(
                camera_id=camera_id,
                time_window_minutes=15
            )
            
            return JsonResponse({
                "status": "processing",
                "message": f"Buffer full ({current_queue_depth} frames). Triggering batch inference + temporal analysis.",
                "camera_id": camera_id
            }, status=202)
        
        return JsonResponse({
            "status": "buffered",
            "message": f"Frame buffered. Queue depth: {current_queue_depth}/5",
            "camera_id": camera_id,
            "queue_percentage": round((current_queue_depth / 5) * 100, 1)
        }, status=202)

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
def get_temporal_sequence(request, camera_id):
    """
    Retrieves the most recent temporal flood sequence analysis for a camera.
    
    GET Parameters:
        - time_window: Time window in minutes (default 15)
    
    Returns:
        JSON with temporal analysis results
    """
    try:
        time_window = int(request.GET.get('time_window', 15))
        
        # Get most recent sequence for this camera
        sequence = TemporalFloodSequence.objects.filter(
            camera__camera_id=camera_id
        ).order_by('-sequence_start').first()
        
        if not sequence:
            return JsonResponse({
                "status": "no_data",
                "message": f"No temporal sequences found for camera {camera_id}",
                "camera_id": camera_id
            }, status=404)
        
        return JsonResponse({
            "status": "success",
            "sequence_id": sequence.id,
            "camera_id": camera_id,
            "num_images": sequence.image_count,
            "time_span_minutes": round((sequence.sequence_end - sequence.sequence_start).total_seconds() / 60, 1),
            "average_depth_cm": sequence.average_depth_cm,
            "max_depth_cm": sequence.max_depth_cm,
            "min_depth_cm": sequence.min_depth_cm,
            "detected_anchor_types": sequence.detected_anchor_types,
            "consensus_water_present": sequence.consensus_water_present,
            "confidence_score": round(sequence.confidence_score, 3),
            "sequence_start": sequence.sequence_start.isoformat(),
            "sequence_end": sequence.sequence_end.isoformat(),
        }, status=200)
        
    except ValueError:
        return JsonResponse({"status": "error", "message": "Invalid time_window parameter"}, status=400)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
def trigger_temporal_analysis(request, camera_id):
    """
    Manually triggers temporal sequence analysis for a camera.
    
    POST Parameters:
        - time_window: Time window in minutes (default 15)
    """
    if request.method != "POST":
        return JsonResponse({"status": "failed"}, status=405)
    
    try:
        time_window = int(request.POST.get('time_window', 15))
        
        # Queue the temporal analysis task
        task = analyze_temporal_sequence.delay(
            camera_id=camera_id,
            time_window_minutes=time_window
        )
        
        return JsonResponse({
            "status": "queued",
            "message": f"Temporal analysis queued for {camera_id} (window: {time_window} min)",
            "task_id": task.id,
            "camera_id": camera_id
        }, status=202)
        
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
def get_camera_stats(request, camera_id):
    """
    Returns statistics for a specific camera.
    
    GET Parameters:
        - hours: Number of hours to analyze (default 24)
    """
    try:
        hours = int(request.GET.get('hours', 24))
        start_time = timezone.now() - timedelta(hours=hours)
        
        camera = CameraLocation.objects.get(camera_id=camera_id)
        
        records = FloodInundationTelemetry.objects.filter(
            camera=camera,
            timestamp__gte=start_time
        )
        
        total_images = records.count()
        water_confirmed = records.filter(is_water_confirmed=True).count()
        avg_depth = records.aggregate(Avg('computed_depth_cm'))['computed_depth_cm__avg']
        max_depth = records.aggregate(Max('computed_depth_cm'))['computed_depth_cm__max']
        
        return JsonResponse({
            "status": "success",
            "camera_id": camera_id,
            "camera_name": camera.location_name,
            "hours_analyzed": hours,
            "total_images": total_images,
            "water_confirmed_images": water_confirmed,
            "avg_depth_cm": round(avg_depth, 2) if avg_depth else 0,
            "max_depth_cm": max_depth,
            "temporal_sequences": TemporalFloodSequence.objects.filter(
                camera=camera,
                sequence_start__gte=start_time
            ).count()
        }, status=200)
        
    except CameraLocation.DoesNotExist:
        return JsonResponse({"status": "error", "message": f"Camera {camera_id} not found"}, status=404)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)