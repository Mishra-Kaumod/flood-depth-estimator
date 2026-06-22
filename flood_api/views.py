# flood_api/views.py
import os
import uuid
import json
import logging
import hashlib
import redis
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg, Max
from django.utils import timezone
from django.db.utils import DatabaseError
from datetime import timedelta
from .models import (
    FloodInundationTelemetry,
    CameraLocation,
    TemporalFloodSequence,
    PredictionFeedback,
    ModelVersion,
    IngestIdempotencyKey,
    FailedTaskEvent,
)
from .services.location_mapping import safe_float, resolve_upload_location
from .services.map_payload import build_dashboard_map_points

# Connect to the Redis instance
redis_client = redis.StrictRedis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
logger = logging.getLogger(__name__)


def _trace_event(level, event, **fields):
    payload = {"event": event, "component": "flood_api.views", **fields}
    log_line = json.dumps(payload, default=str)
    if level == "error":
        logger.error(log_line)
    elif level == "warning":
        logger.warning(log_line)
    else:
        logger.info(log_line)


def _get_idempotency_key(request):
    header_key = request.META.get("HTTP_IDEMPOTENCY_KEY", "").strip()
    body_key = request.POST.get("idempotency_key", "").strip() if request.POST else ""
    return header_key or body_key or None


def _idempotency_cache_lookup(endpoint, idempotency_key):
    if not idempotency_key:
        return None
    ttl_cutoff = timezone.now() - timedelta(hours=settings.INGEST_IDEMPOTENCY_TTL_HOURS)
    IngestIdempotencyKey.objects.filter(created_at__lt=ttl_cutoff).delete()
    return IngestIdempotencyKey.objects.filter(endpoint=endpoint, key=idempotency_key).first()


def _idempotency_cache_store(endpoint, idempotency_key, response_payload, response_status):
    if not idempotency_key:
        return
    IngestIdempotencyKey.objects.update_or_create(
        endpoint=endpoint,
        key=idempotency_key,
        defaults={
            "response_status": int(response_status),
            "response_payload": response_payload,
        },
    )


def _publish_dlq(task, payload, error_message, retry_count, source_endpoint):
    event_payload = {
        "task": task.name,
        "source_endpoint": source_endpoint,
        "payload_fingerprint": hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "error": error_message,
        "retry_count": retry_count,
        "created_at": timezone.now().isoformat(),
    }
    FailedTaskEvent.objects.create(
        task_name=task.name,
        source_endpoint=source_endpoint or "",
        payload=payload,
        error_message=error_message,
        retry_count=retry_count,
    )
    if _celery_backend_available():
        try:
            redis_client.lpush(settings.TASK_DLQ_REDIS_KEY, json.dumps(event_payload))
            redis_client.ltrim(settings.TASK_DLQ_REDIS_KEY, 0, 1000)
        except Exception as exc:
            _trace_event("warning", "dlq_redis_push_failed", error=str(exc), source_endpoint=source_endpoint)
    _trace_event(
        "error",
        "task_routed_to_dlq",
        task_name=task.name,
        source_endpoint=source_endpoint,
        retry_count=retry_count,
        error=error_message,
    )


def _celery_backend_available():
    try:
        redis_client.ping()
        return True
    except Exception as exc:
        logger.warning("Celery/Redis backend unavailable: %s", exc)
        return False


def _run_inline_with_retries(task, source_endpoint="", **kwargs):
    max_retries = max(0, int(settings.INLINE_TASK_MAX_RETRIES))
    for attempt in range(max_retries + 1):
        try:
            task.run(**kwargs)
            return {"mode": "inline", "task_id": None}
        except Exception as exc:
            _trace_event(
                "warning",
                "inline_task_attempt_failed",
                task_name=task.name,
                source_endpoint=source_endpoint,
                attempt=attempt + 1,
                max_attempts=max_retries + 1,
                error=str(exc),
            )
            if attempt >= max_retries:
                _publish_dlq(
                    task=task,
                    payload=kwargs,
                    error_message=str(exc),
                    retry_count=max_retries,
                    source_endpoint=source_endpoint,
                )
                return {"mode": "inline_failed", "task_id": None, "error": str(exc)}


def _dispatch_task_with_fallback(task, source_endpoint="", **kwargs):
    """
    Try Celery async first; if broker/result backend is unavailable,
    run the task inline so local dashboard uploads still work.
    """
    if not _celery_backend_available():
        return _run_inline_with_retries(task, source_endpoint=source_endpoint, **kwargs)

    try:
        async_result = task.delay(**kwargs)
        return {"mode": "queued", "task_id": getattr(async_result, "id", None)}
    except Exception as exc:
        _trace_event("warning", "celery_queue_failed_falling_back_inline", task_name=task.name, error=str(exc))
        return _run_inline_with_retries(task, source_endpoint=source_endpoint, **kwargs)

def dashboard_view(request):
    """
    Renders the web interface. Manual uploads here bypass the 
    batch-buffer and go straight to the ML/LLM worker.
    """
    context = {}
    if request.method == "POST" and request.FILES.getlist("image_file"):
        try:
            uploaded_files = request.FILES.getlist("image_file")
            base_camera_id = request.POST.get("camera_id", "").strip()
            base_location_id = request.POST.get("location_id", "").strip()
            location_name = request.POST.get("location_name", "").strip()
            latitude = safe_float(request.POST.get("latitude"))
            longitude = safe_float(request.POST.get("longitude"))

            for index, uploaded_file in enumerate(uploaded_files):
                if base_camera_id:
                    camera_id = base_camera_id if len(uploaded_files) == 1 else f"{base_camera_id}_{index + 1}"
                else:
                    camera_id = f"manual_{uuid.uuid4().hex[:8]}"

                (
                    resolved_location_name,
                    resolved_latitude,
                    resolved_longitude,
                ) = resolve_upload_location(
                    location_name=location_name,
                    latitude=latitude,
                    longitude=longitude,
                    fallback_seed=f"{camera_id}:{uploaded_file.name}",
                )

                camera, created = CameraLocation.objects.get_or_create(
                    camera_id=camera_id,
                    defaults={
                        "location_id": base_location_id or None,
                        "location_name": resolved_location_name,
                        "latitude": resolved_latitude,
                        "longitude": resolved_longitude,
                        "description": "Web dashboard upload camera mapping",
                    },
                )
                if not created:
                    camera_updated = False
                    if camera.latitude is None:
                        camera.latitude = resolved_latitude
                        camera_updated = True
                    if camera.longitude is None:
                        camera.longitude = resolved_longitude
                        camera_updated = True
                    if not camera.location_name:
                        camera.location_name = resolved_location_name
                        camera_updated = True
                    if not camera.location_id and base_location_id:
                        camera.location_id = base_location_id
                        camera_updated = True
                    if camera_updated:
                        camera.save(update_fields=["latitude", "longitude", "location_name", "location_id"])

                # Save file temporarily for the background worker
                temp_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
                temp_path = os.path.join(str(settings.RUNTIME_TMP_DIR), temp_filename)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)

                with open(temp_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                from .tasks import process_and_refine_telemetry
                dispatch_info = _dispatch_task_with_fallback(
                    process_and_refine_telemetry,
                    source_endpoint="dashboard_view",
                    image_filepath=temp_path,
                    filename=uploaded_file.name,
                    external_context="Manual Web Dashboard Upload",
                    camera_id=camera.camera_id
                )
                if dispatch_info["mode"] == "inline_failed":
                    raise RuntimeError(f"Task failed for {uploaded_file.name}: {dispatch_info.get('error', 'unknown')}")

            context["success"] = True
            if dispatch_info["mode"] == "inline":
                context["message"] = (
                    f"{len(uploaded_files)} image(s) processed inline "
                    "(Celery backend unavailable)."
                )
            else:
                context["message"] = (
                    f"{len(uploaded_files)} image(s) accepted. "
                    "AI workers are processing them in the background. "
                    "Refresh shortly to see mapped flood points."
                )
        except Exception as e:
            context["error"] = f"Failed to process file: {str(e)}"

    try:
        records = FloodInundationTelemetry.objects.select_related("camera").all()[:50]
        map_points = build_dashboard_map_points(records)
    except DatabaseError:
        records = []
        map_points = []

    # Pull the latest telemetry from the database for the UI table
    context["historical_records"] = records[:20]
    try:
        context["cameras"] = CameraLocation.objects.all()
    except DatabaseError:
        context["cameras"] = []
    context["map_points_json"] = json.dumps(map_points)
    
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
        idempotency_key = _get_idempotency_key(request)
        cached = _idempotency_cache_lookup("high_speed_api_endpoint", idempotency_key)
        if cached:
            return JsonResponse(cached.response_payload, status=cached.response_status)

        uploaded_file = request.FILES.get("image")
        camera_id = request.POST.get("camera_id", "intersection_01")
        location_id = request.POST.get("location_id", "").strip()
        location_name = request.POST.get("location_name", f"Location {camera_id}")
        latitude = safe_float(request.POST.get("latitude", None))
        longitude = safe_float(request.POST.get("longitude", None))
        external_context = request.POST.get("context", "")
        
        if not uploaded_file:
            return JsonResponse({"status": "failed", "error": "No image payload"}, status=400)

        # Create or update camera location
        camera, created = CameraLocation.objects.get_or_create(
            camera_id=camera_id,
            defaults={
                'location_id': location_id or None,
                'location_name': location_name,
                'latitude': latitude,
                'longitude': longitude,
            }
        )
        
        temp_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        temp_path = os.path.join(str(settings.RUNTIME_TMP_DIR), temp_filename)
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
            from .tasks import process_and_refine_telemetry, analyze_temporal_sequence
            inference_dispatch = _dispatch_task_with_fallback(
                process_and_refine_telemetry,
                source_endpoint="high_speed_api_endpoint",
                image_filepath=latest_payload["image_filepath"],
                filename=latest_payload["filename"],
                external_context=latest_payload["external_context"],
                camera_id=latest_payload["camera_id"]
            )
            
            # Trigger temporal analysis on sequence
            temporal_dispatch = _dispatch_task_with_fallback(
                analyze_temporal_sequence,
                source_endpoint="high_speed_api_endpoint",
                camera_id=camera_id,
                time_window_minutes=15
            )
            if inference_dispatch["mode"] == "inline_failed" or temporal_dispatch["mode"] == "inline_failed":
                response_payload = {
                    "status": "failed",
                    "message": "Inline processing failed after retries",
                    "camera_id": camera_id,
                    "inference_mode": inference_dispatch["mode"],
                    "temporal_mode": temporal_dispatch["mode"],
                }
                _idempotency_cache_store("high_speed_api_endpoint", idempotency_key, response_payload, 500)
                return JsonResponse(response_payload, status=500)
            
            response_payload = {
                "status": "processing",
                "message": f"Buffer full ({current_queue_depth} frames). Triggering batch inference + temporal analysis.",
                "camera_id": camera_id,
                "inference_mode": inference_dispatch["mode"],
                "temporal_mode": temporal_dispatch["mode"],
            }
            _idempotency_cache_store("high_speed_api_endpoint", idempotency_key, response_payload, 202)
            return JsonResponse(response_payload, status=202)
        
        response_payload = {
            "status": "buffered",
            "message": f"Frame buffered. Queue depth: {current_queue_depth}/5",
            "camera_id": camera_id,
            "queue_percentage": round((current_queue_depth / 5) * 100, 1)
        }
        _idempotency_cache_store("high_speed_api_endpoint", idempotency_key, response_payload, 202)
        return JsonResponse(response_payload, status=202)

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
def health_live(request):
    return JsonResponse({"status": "ok", "service": "flood-api", "check": "live"}, status=200)


@csrf_exempt
def health_ready(request):
    db_ok = True
    redis_ok = True
    db_error = ""
    redis_error = ""
    dlq_depth = 0
    failed_tasks_last_hour = 0
    try:
        CameraLocation.objects.first()
        failed_tasks_last_hour = FailedTaskEvent.objects.filter(
            created_at__gte=timezone.now() - timedelta(hours=1),
            resolved=False,
        ).count()
    except Exception as exc:
        db_ok = False
        db_error = str(exc)
    try:
        redis_client.ping()
        dlq_depth = int(redis_client.llen(settings.TASK_DLQ_REDIS_KEY))
    except Exception as exc:
        redis_ok = False
        redis_error = str(exc)

    code = 200 if db_ok else 503
    return JsonResponse(
        {
            "status": "ok" if code == 200 else "degraded",
            "service": "flood-api",
            "checks": {
                "database": {"ok": db_ok, "error": db_error},
                "redis": {"ok": redis_ok, "error": redis_error},
                "dlq_depth": dlq_depth,
                "failed_tasks_last_hour": failed_tasks_last_hour,
            },
            "alerts": {
                "failed_tasks_spike": failed_tasks_last_hour >= settings.OPS_ALERT_FAILED_TASKS_PER_HOUR,
                "dlq_depth_high": dlq_depth >= settings.OPS_ALERT_DLQ_DEPTH if redis_ok else False,
            },
        },
        status=code,
    )


@csrf_exempt
def ops_metrics_api(request):
    unresolved_failures = FailedTaskEvent.objects.filter(resolved=False).count()
    failed_last_hour = FailedTaskEvent.objects.filter(
        created_at__gte=timezone.now() - timedelta(hours=1),
        resolved=False,
    ).count()
    redis_ok = _celery_backend_available()
    dlq_depth = 0
    if redis_ok:
        try:
            dlq_depth = int(redis_client.llen(settings.TASK_DLQ_REDIS_KEY))
        except Exception as exc:
            _trace_event("warning", "ops_metrics_dlq_read_failed", error=str(exc))
    payload = {
        "status": "success",
        "metrics": {
            "failed_tasks_unresolved": unresolved_failures,
            "failed_tasks_last_hour": failed_last_hour,
            "idempotency_records_24h": IngestIdempotencyKey.objects.filter(
                created_at__gte=timezone.now() - timedelta(hours=24)
            ).count(),
            "redis_available": redis_ok,
            "dlq_depth": dlq_depth,
        },
        "alerts": {
            "failed_tasks_spike": failed_last_hour >= settings.OPS_ALERT_FAILED_TASKS_PER_HOUR,
            "dlq_depth_high": dlq_depth >= settings.OPS_ALERT_DLQ_DEPTH if redis_ok else False,
        },
    }
    return JsonResponse(payload, status=200)


def _serialize_telemetry(record):
    return {
        "id": record.id,
        "timestamp": record.timestamp.isoformat(),
        "camera_id": record.camera.camera_id if record.camera else None,
        "location_id": record.camera.location_id if record.camera else None,
        "location_name": record.camera.location_name if record.camera else None,
        "latitude": record.camera.latitude if record.camera else None,
        "longitude": record.camera.longitude if record.camera else None,
        "image_name": record.image_name,
        "strategy_applied": record.strategy_applied,
        "surface_water_confirmed_pct": record.surface_water_confirmed_pct,
        "computed_depth_cm": record.computed_depth_cm,
        "system_confidence_score_pct": record.system_confidence_score_pct,
        "detected_reference_objects": record.detected_reference_objects,
        "num_reference_objects": record.num_reference_objects,
        "is_water_confirmed": record.is_water_confirmed,
        "safety_risk_assessment": record.safety_risk_assessment,
    }


@csrf_exempt
def batch_ingest_api(request):
    if request.method != "POST":
        return JsonResponse({"status": "failed", "error": "Method not allowed"}, status=405)

    idempotency_key = _get_idempotency_key(request)
    cached = _idempotency_cache_lookup("batch_ingest_api", idempotency_key)
    if cached:
        return JsonResponse(cached.response_payload, status=cached.response_status)

    uploaded_files = request.FILES.getlist("images")
    if not uploaded_files:
        fallback = request.FILES.get("image")
        if fallback:
            uploaded_files = [fallback]

    if not uploaded_files:
        return JsonResponse({"status": "failed", "error": "No images uploaded"}, status=400)

    base_camera_id = request.POST.get("camera_id", "").strip()
    base_location_id = request.POST.get("location_id", "").strip()
    location_name = request.POST.get("location_name", "").strip()
    latitude = safe_float(request.POST.get("latitude"))
    longitude = safe_float(request.POST.get("longitude"))
    external_context = request.POST.get("context", "API batch upload")

    execution_modes = []
    accepted = []
    failed = []

    for index, uploaded_file in enumerate(uploaded_files):
        if base_camera_id:
            camera_id = base_camera_id if len(uploaded_files) == 1 else f"{base_camera_id}_{index + 1}"
        else:
            camera_id = f"api_{uuid.uuid4().hex[:8]}"

        resolved_location_name, resolved_latitude, resolved_longitude = resolve_upload_location(
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            fallback_seed=f"{camera_id}:{uploaded_file.name}",
        )

        camera, created = CameraLocation.objects.get_or_create(
            camera_id=camera_id,
            defaults={
                "location_id": base_location_id or None,
                "location_name": resolved_location_name,
                "latitude": resolved_latitude,
                "longitude": resolved_longitude,
                "description": "API batch ingest camera mapping",
            },
        )
        if not created:
            camera_updated = False
            if camera.latitude is None:
                camera.latitude = resolved_latitude
                camera_updated = True
            if camera.longitude is None:
                camera.longitude = resolved_longitude
                camera_updated = True
            if not camera.location_name:
                camera.location_name = resolved_location_name
                camera_updated = True
            if not camera.location_id and base_location_id:
                camera.location_id = base_location_id
                camera_updated = True
            if camera_updated:
                camera.save(update_fields=["latitude", "longitude", "location_name", "location_id"])

        temp_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        temp_path = os.path.join(str(settings.RUNTIME_TMP_DIR), temp_filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb+") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        from .tasks import process_and_refine_telemetry
        dispatch_info = _dispatch_task_with_fallback(
            process_and_refine_telemetry,
            source_endpoint="batch_ingest_api",
            image_filepath=temp_path,
            filename=uploaded_file.name,
            external_context=external_context,
            camera_id=camera.camera_id,
        )
        execution_modes.append(dispatch_info["mode"])
        if dispatch_info["mode"] == "inline_failed":
            failed.append(
                {
                    "file_name": uploaded_file.name,
                    "camera_id": camera.camera_id,
                    "error": dispatch_info.get("error", "unknown"),
                }
            )
            continue
        accepted.append(
            {
                "file_name": uploaded_file.name,
                "camera_id": camera.camera_id,
                "location_id": camera.location_id,
                "location_name": camera.location_name,
                "latitude": camera.latitude,
                "longitude": camera.longitude,
                "execution_mode": dispatch_info["mode"],
            }
        )

    mode = "inline" if "inline" in execution_modes else "queued"
    status_code = 202 if not failed else 207
    response_payload = {
        "status": "accepted" if not failed else "partial_failure",
        "count": len(accepted),
        "failed_count": len(failed),
        "execution_mode": mode,
        "items": accepted,
        "failed_items": failed,
    }
    _idempotency_cache_store("batch_ingest_api", idempotency_key, response_payload, status_code)
    return JsonResponse(response_payload, status=status_code)


@csrf_exempt
def telemetry_map_points_api(request):
    limit = int(request.GET.get("limit", 200))
    records = FloodInundationTelemetry.objects.select_related("camera").all()[:limit]
    points = build_dashboard_map_points(records)
    return JsonResponse({"status": "success", "count": len(points), "points": points}, status=200)


@csrf_exempt
def telemetry_detail_api(request, telemetry_id):
    try:
        record = FloodInundationTelemetry.objects.select_related("camera").get(id=telemetry_id)
    except FloodInundationTelemetry.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Telemetry not found"}, status=404)
    return JsonResponse({"status": "success", "telemetry": _serialize_telemetry(record)}, status=200)


@csrf_exempt
def feedback_submit_api(request):
    if request.method != "POST":
        return JsonResponse({"status": "failed", "error": "Method not allowed"}, status=405)

    payload = request.POST if request.POST else json.loads(request.body.decode("utf-8"))
    telemetry_id = payload.get("telemetry_id")
    feedback_type = payload.get("feedback_type")
    if not telemetry_id or not feedback_type:
        return JsonResponse(
            {"status": "error", "message": "telemetry_id and feedback_type are required"},
            status=400,
        )

    try:
        telemetry = FloodInundationTelemetry.objects.get(id=telemetry_id)
    except FloodInundationTelemetry.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Telemetry not found"}, status=404)

    feedback = PredictionFeedback.objects.create(
        telemetry=telemetry,
        feedback_type=feedback_type,
        corrected_depth_cm=safe_float(payload.get("corrected_depth_cm")),
        corrected_risk=payload.get("corrected_risk", ""),
        reviewer=payload.get("reviewer", ""),
        notes=payload.get("notes", ""),
        metadata={
            "source": payload.get("source", "api"),
            "location_id": payload.get("location_id"),
        },
    )
    return JsonResponse({"status": "success", "feedback_id": feedback.id}, status=201)


@csrf_exempt
def feedback_queue_api(request):
    threshold = safe_float(request.GET.get("confidence_below"))
    if threshold is None:
        threshold = 65.0
    limit = int(request.GET.get("limit", 100))
    records = (
        FloodInundationTelemetry.objects.select_related("camera")
        .filter(system_confidence_score_pct__lt=threshold)
        .order_by("-timestamp")[:limit]
    )
    return JsonResponse(
        {"status": "success", "count": len(records), "items": [_serialize_telemetry(r) for r in records]},
        status=200,
    )


@csrf_exempt
def model_versions_api(request):
    if request.method == "GET":
        versions = ModelVersion.objects.all()[:200]
        items = [
            {
                "id": v.id,
                "model_name": v.model_name,
                "version": v.version,
                "stage": v.stage,
                "metadata": v.metadata,
                "created_at": v.created_at.isoformat(),
            }
            for v in versions
        ]
        return JsonResponse({"status": "success", "count": len(items), "items": items}, status=200)

    if request.method != "POST":
        return JsonResponse({"status": "failed", "error": "Method not allowed"}, status=405)

    payload = request.POST if request.POST else json.loads(request.body.decode("utf-8"))
    model_name = payload.get("model_name")
    version = payload.get("version")
    if not model_name or not version:
        return JsonResponse({"status": "error", "message": "model_name and version are required"}, status=400)
    model = ModelVersion.objects.create(
        model_name=model_name,
        version=version,
        stage=payload.get("stage", "candidate"),
        metadata=payload.get("metadata", {}),
    )
    return JsonResponse({"status": "success", "id": model.id}, status=201)


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
        from .tasks import analyze_temporal_sequence
        # Queue the temporal analysis task
        dispatch_info = _dispatch_task_with_fallback(
            analyze_temporal_sequence,
            camera_id=camera_id,
            time_window_minutes=time_window
        )
        
        return JsonResponse({
            "status": "queued",
            "message": f"Temporal analysis queued for {camera_id} (window: {time_window} min)",
            "task_id": dispatch_info["task_id"],
            "execution_mode": dispatch_info["mode"],
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


# ============================================================================
# ACTIVE LEARNING & RETRAINING ENDPOINTS
# ============================================================================

@csrf_exempt
def verify_prediction(request):
    """
    Accept ground-truth corrections from human operators.
    Endpoint: POST /api/v1/floods/verify/
    
    Request body:
    {
        "telemetry_id": "uuid",
        "feedback_type": "rejected|corrected|accepted",
        "corrected_flood": 0 or 1 (optional),
        "corrected_depth_cm": 25.5 (optional),
        "corrected_risk": "string" (optional),
        "reviewer": "operator_name",
        "notes": "explanation",
        "scene_conditions": {
            "time_of_day": "morning|afternoon|night",
            "weather": "clear|rain|overcast",
            "occlusion": "none|partial|heavy"
        }
    }
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)

    try:
        if request.content_type == "application/json":
            payload = json.loads(request.body)
        else:
            payload = request.POST.dict()
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"status": "error", "message": "Invalid JSON"}, status=400)

    # Validate required fields
    telemetry_id = payload.get("telemetry_id")
    feedback_type = payload.get("feedback_type", "corrected")

    if not telemetry_id:
        return JsonResponse(
            {"status": "error", "message": "telemetry_id is required"},
            status=400,
        )

    try:
        telemetry = FloodInundationTelemetry.objects.get(id=telemetry_id)
    except FloodInundationTelemetry.DoesNotExist:
        return JsonResponse(
            {"status": "error", "message": f"Telemetry {telemetry_id} not found"},
            status=404,
        )

    # Create feedback record
    feedback = PredictionFeedback.objects.create(
        telemetry=telemetry,
        feedback_type=feedback_type,
        corrected_depth_cm=safe_float(payload.get("corrected_depth_cm")),
        corrected_risk=payload.get("corrected_risk", ""),
        reviewer=payload.get("reviewer", "operator"),
        notes=payload.get("notes", ""),
        metadata={
            "source": "api_verify",
            "corrected_flood": payload.get("corrected_flood"),
            "scene_conditions": payload.get("scene_conditions", {}),
        },
    )

    _trace_event(
        "info",
        "feedback_received",
        telemetry_id=str(telemetry_id),
        feedback_id=str(feedback.id),
        feedback_type=feedback_type,
        reviewer=feedback.reviewer,
    )

    # Check if retraining should trigger
    from flood_api.ml_ops.retraining_trigger import RetrainingTrigger

    trigger = RetrainingTrigger()
    should_retrain, reason, metadata = trigger.check_trigger()

    response = {
        "status": "success",
        "feedback_id": str(feedback.id),
        "telemetry_id": str(telemetry_id),
        "feedback_type": feedback_type,
        "retrain_status": "triggered" if should_retrain else "monitoring",
        "retrain_reason": reason if should_retrain else None,
    }

    return JsonResponse(response, status=201)


@csrf_exempt
def feedback_summary_api(request):
    """
    Get summary of feedback corrections and retraining status.
    Endpoint: GET /api/v1/floods/feedback-summary/
    """
    days = int(request.GET.get("days", 7))
    cutoff = timezone.now() - timedelta(days=days)

    total = PredictionFeedback.objects.filter(created_at__gte=cutoff).count()
    rejected = PredictionFeedback.objects.filter(
        created_at__gte=cutoff, feedback_type="rejected"
    ).count()
    corrected = PredictionFeedback.objects.filter(
        created_at__gte=cutoff, feedback_type="corrected"
    ).count()
    accepted = PredictionFeedback.objects.filter(
        created_at__gte=cutoff, feedback_type="accepted"
    ).count()

    used_in_training = PredictionFeedback.objects.filter(
        created_at__gte=cutoff, metadata__has_key="used_in_training"
    ).count()

    current_prod = ModelVersion.objects.filter(stage="production").first()
    staging = ModelVersion.objects.filter(stage="staging").first()

    return JsonResponse(
        {
            "status": "success",
            "days": days,
            "feedback_summary": {
                "total": total,
                "rejected": rejected,
                "corrected": corrected,
                "accepted": accepted,
                "used_in_training": used_in_training,
                "pending": total - used_in_training,
            },
            "model_versions": {
                "production": {
                    "version": current_prod.version if current_prod else None,
                    "checkpoint": current_prod.checkpoint_path if current_prod else None,
                },
                "staging": {
                    "version": staging.version if staging else None,
                    "checkpoint": staging.checkpoint_path if staging else None,
                },
            },
        },
        status=200,
    )


@csrf_exempt
def retrain_trigger_manual(request):
    """
    Manually trigger retraining (for testing/ops).
    Endpoint: POST /api/v1/ml-ops/retrain-trigger-manual/
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)

    try:
        from flood_api.ml_ops.retraining_trigger import RetrainingTrigger

        trigger = RetrainingTrigger()

        # Check trigger conditions
        should_retrain, reason, metadata = trigger.check_trigger()

        if not should_retrain:
            return JsonResponse(
                {
                    "status": "not_triggered",
                    "message": reason,
                    "metadata": metadata,
                },
                status=200,
            )

        # Run retraining
        success, new_version, message = trigger.retrain_and_evaluate()

        if success:
            return JsonResponse(
                {
                    "status": "success",
                    "message": message,
                    "new_version": new_version.version,
                    "stage": new_version.stage,
                    "checkpoint": new_version.checkpoint_path,
                },
                status=200,
            )
        else:
            return JsonResponse(
                {
                    "status": "failed",
                    "message": message,
                },
                status=400,
            )

    except Exception as e:
        logger.exception("Manual retrain trigger failed")
        return JsonResponse(
            {
                "status": "error",
                "message": str(e),
            },
            status=500,
        )


@csrf_exempt
def model_promotion_api(request):
    """
    Promote a model from staging to production.
    Endpoint: POST /api/v1/ml-ops/model-promotion/
    
    Request body:
    {
        "staging_version": "1.1",
        "promoted_by": "ops_team"
    }
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)

    try:
        if request.content_type == "application/json":
            payload = json.loads(request.body)
        else:
            payload = request.POST.dict()

        staging_version = payload.get("staging_version")
        promoted_by = payload.get("promoted_by", "api")

        if not staging_version:
            return JsonResponse(
                {"status": "error", "message": "staging_version is required"},
                status=400,
            )

        # Find staging model
        staging = ModelVersion.objects.get(version=staging_version, stage="staging")

        # Demote current production
        current_prod = ModelVersion.objects.filter(stage="production").first()
        if current_prod:
            current_prod.stage = "archived"
            current_prod.save()
            logger.info(f"Archived previous production version: {current_prod.version}")

        # Promote staging to production
        staging.stage = "production"
        staging.metadata["promoted_by"] = promoted_by
        staging.metadata["promoted_at"] = timezone.now().isoformat()
        staging.save()

        _trace_event(
            "info",
            "model_promotion",
            from_version=staging_version,
            to_stage="production",
            promoted_by=promoted_by,
        )

        return JsonResponse(
            {
                "status": "success",
                "message": f"Model {staging_version} promoted to production",
                "version": staging.version,
                "stage": staging.stage,
            },
            status=200,
        )

    except ModelVersion.DoesNotExist:
        return JsonResponse(
            {
                "status": "error",
                "message": f"Staging version {staging_version} not found",
            },
            status=404,
        )
    except Exception as e:
        logger.exception("Model promotion failed")
        return JsonResponse(
            {
                "status": "error",
                "message": str(e),
            },
            status=500,
        )