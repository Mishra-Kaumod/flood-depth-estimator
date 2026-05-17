# flood_api/views.py
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
import cv2
import numpy as np
import tempfile
import os
from core_logic import estimate_flood_depth

def upload_ui(request):
    """
    Renders the browser web page for manual front-end interaction.
    """
    return render(request, 'upload.html')

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def analyze_media(request):
    """
    Unified Endpoint: Accepts images or videos via standard API payloads 
    or front-end browser form submissions.
    """
    # -------------------------------------------------------------
    # CASE 1: MULTIPLE IMAGES BATCH PROCESSING
    # -------------------------------------------------------------
    if 'images' in request.FILES:
        files = request.FILES.getlist('images')
        batch_results = []
        
        for idx, uploaded_file in enumerate(files):
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image_matrix = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image_matrix is not None:
                analysis = estimate_flood_depth(image_matrix)
                analysis['file_name'] = uploaded_file.name
                batch_results.append(analysis)
                
        return JsonResponse({"status": "success", "mode": "batch_images", "results": batch_results})

    # -------------------------------------------------------------
    # CASE 2: SINGLE VIDEO PROCESSING (Extracts Key Frames)
    # -------------------------------------------------------------
    elif 'video' in request.FILES:
        uploaded_video = request.FILES['video']
        
        # Save uploaded video stream to a temporary local file for OpenCV VideoCapture
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as temp_video:
            for chunk in uploaded_video.chunks():
                temp_video.write(chunk)
            temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        frame_idx = 0
        video_depths = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # To optimize processing speed, sample 1 frame every 1 second (Assuming 30 FPS video)
            if frame_idx % 30 == 0:
                analysis = estimate_flood_depth(frame)
                video_depths.append({
                    "timestamp_seconds": round(frame_idx / 30, 2),
                    "estimated_depth_cm": analysis["estimated_depth_cm"],
                    "anchors_found": analysis["anchors_detected"]
                })
            frame_idx += 1
            
        cap.release()
        os.unlink(temp_video_path) # Safe deletion of temporary video container
        
        # Calculate summary statistics across the video timeline
        all_depths = [v["estimated_depth_cm"] for v in video_depths if v["estimated_depth_cm"] > 0]
        peak_depth = max(all_depths) if all_depths else 0.0
        
        return JsonResponse({
            "status": "success",
            "mode": "video",
            "file_name": uploaded_video.name,
            "peak_depth_detected_cm": peak_depth,
            "timeline_analysis": video_depths
        })

    # -------------------------------------------------------------
    # CASE 3: STANDARD SINGLE IMAGE UPLOAD
    # -------------------------------------------------------------
    elif 'image' in request.FILES:
        uploaded_file = request.FILES['image']
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_matrix = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image_matrix is None:
            return JsonResponse({"status": "error", "message": "Failed to decode image context."}, status=400)
            
        results = estimate_flood_depth(image_matrix)
        results['file_name'] = uploaded_file.name
        return JsonResponse(results)

    return JsonResponse({"status": "error", "message": "No valid keys found. Please supply 'image', 'images', or 'video'."}, status=400)

# Create your views here.
