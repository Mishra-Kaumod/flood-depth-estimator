"""
Batch Image Upload View & Forms for Flood Detection
Supports uploading 10 images at once with color-coded results.
"""

from django import forms
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.db import models
import json
from pathlib import Path
import numpy as np

class BatchImageUploadForm(forms.Form):
    """Form for batch uploading up to 10 images."""
    
    images = forms.FileField(
        widget=forms.ClearableFileInput(attrs={
            'multiple': True,
            'accept': 'image/*',
            'class': 'form-control'
        }),
        label='Upload 10 Test Images (JPEG, PNG)',
        required=True
    )
    
    scenario_name = forms.CharField(
        max_length=100,
        initial='Bengaluru - June 2026 Monsoon Testing',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Bengaluru Monsoon Test'
        })
    )
    
    location = forms.CharField(
        max_length=200,
        initial='Bengaluru, India - Multiple Sites',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Indiranagar, Koramangala, Whitefield'
        })
    )
    
    description = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional: Notes about the test scenario'
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        files = self.files.getlist('images')
        
        if not files:
            raise forms.ValidationError("Please select at least 1 image.")
        
        if len(files) > 10:
            raise forms.ValidationError("Maximum 10 images allowed per batch.")
        
        for f in files:
            if not f.content_type.startswith('image/'):
                raise forms.ValidationError(f"File {f.name} is not an image.")
            
            if f.size > 10 * 1024 * 1024:  # 10MB max
                raise forms.ValidationError(f"File {f.name} exceeds 10MB limit.")
        
        return cleaned_data


class BatchUploadResult(models.Model):
    """Model to store batch upload results."""
    
    SCENARIO_CHOICES = [
        ('monsoon', 'Monsoon Season'),
        ('urban', 'Urban Area'),
        ('rural', 'Rural Area'),
        ('edge_case', 'Edge Cases'),
        ('other', 'Other'),
    ]
    
    batch_id = models.CharField(max_length=50, unique=True)
    scenario_name = models.CharField(max_length=100)
    location = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    total_images = models.IntegerField()
    flooded_count = models.IntegerField(default=0)
    dry_count = models.IntegerField(default=0)
    
    avg_confidence = models.FloatField(default=0.0)
    avg_depth_cm = models.FloatField(default=0.0)
    
    results_json = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.batch_id} - {self.scenario_name}"


@require_http_methods(["POST"])
def batch_upload_images(request):
    """
    Handle batch upload of up to 10 images.
    Process with flood detection model.
    Return color-coded results.
    """
    from flood_api.models import FloodInundationTelemetry
    import tempfile
    from django.utils import timezone
    from django.conf import settings
    
    # Validate form
    form = BatchImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({
            'status': 'error',
            'errors': form.errors
        }, status=400)
    
    # Get files
    files = request.FILES.getlist('images')
    scenario_name = form.cleaned_data['scenario_name']
    location = form.cleaned_data['location']
    description = form.cleaned_data['description']
    
    # Generate batch ID
    batch_id = f"batch_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
    
    results = []
    flooded_count = 0
    dry_count = 0
    
    try:
        # Process each image
        for idx, uploaded_file in enumerate(files, 1):
            try:
                # Load image
                import cv2
                from PIL import Image
                import io
                
                img_data = uploaded_file.read()
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Run inference
                from .inference import FloodDetectionInference
                
                inference = FloodDetectionInference()
                pred = inference.predict_single(img)
                
                # Determine intensity color
                confidence = pred['yolo_confidence']
                is_flood = pred['flood_predicted']
                depth = pred.get('llama_depth_cm', 0)
                
                # Color coding based on confidence & flood status
                if is_flood:
                    if confidence > 0.9:
                        color_code = '#8B0000'  # Dark red (critical)
                        intensity = 'CRITICAL'
                    elif confidence > 0.75:
                        color_code = '#FF4500'  # Orange red (high)
                        intensity = 'HIGH'
                    else:
                        color_code = '#FFD700'  # Gold (medium)
                        intensity = 'MEDIUM'
                    flooded_count += 1
                else:
                    if confidence > 0.8:
                        color_code = '#228B22'  # Forest green (safe)
                        intensity = 'SAFE'
                    else:
                        color_code = '#FFD700'  # Gold (uncertain)
                        intensity = 'UNCERTAIN'
                    dry_count += 1
                
                result = {
                    'image_num': idx,
                    'filename': uploaded_file.name,
                    'is_flood': is_flood,
                    'confidence': round(confidence, 3),
                    'depth_cm': round(depth, 2),
                    'color_code': color_code,
                    'intensity': intensity,
                    'status': 'success'
                }
                
                results.append(result)
                
                # Save to database
                telemetry = FloodInundationTelemetry.objects.create(
                    batch_id=batch_id,
                    location=location,
                    yolo_confidence=confidence,
                    yolo_flood_prediction=is_flood,
                    llama_depth_cm=depth,
                    image_path=f"batch_uploads/{batch_id}/{uploaded_file.name}",
                    inference_metadata={
                        'intensity': intensity,
                        'color_code': color_code
                    }
                )
                
            except Exception as e:
                results.append({
                    'image_num': idx,
                    'filename': uploaded_file.name,
                    'status': 'error',
                    'error': str(e),
                    'color_code': '#808080',
                    'intensity': 'ERROR'
                })
        
        # Calculate batch statistics
        total = len(results)
        flooded = sum(1 for r in results if r.get('is_flood'))
        dry = sum(1 for r in results if not r.get('is_flood', False) and r.get('status') == 'success')
        
        avg_conf = sum(r.get('confidence', 0) for r in results if r.get('status') == 'success') / max(total, 1)
        avg_depth = sum(r.get('depth_cm', 0) for r in results if r.get('status') == 'success') / max(total, 1)
        
        # Save batch result
        batch_result = BatchUploadResult.objects.create(
            batch_id=batch_id,
            scenario_name=scenario_name,
            location=location,
            description=description,
            total_images=total,
            flooded_count=flooded,
            dry_count=dry,
            avg_confidence=avg_conf,
            avg_depth_cm=avg_depth,
            results_json={'results': results}
        )
        
        return JsonResponse({
            'status': 'success',
            'batch_id': batch_id,
            'scenario': scenario_name,
            'location': location,
            'total_images': total,
            'flooded_count': flooded,
            'dry_count': dry,
            'avg_confidence': round(avg_conf, 3),
            'avg_depth_cm': round(avg_depth, 2),
            'results': results
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


def get_batch_results(request, batch_id):
    """Retrieve batch results for display."""
    try:
        batch = BatchUploadResult.objects.get(batch_id=batch_id)
        return JsonResponse({
            'status': 'success',
            'batch': {
                'batch_id': batch.batch_id,
                'scenario_name': batch.scenario_name,
                'location': batch.location,
                'total_images': batch.total_images,
                'flooded_count': batch.flooded_count,
                'dry_count': batch.dry_count,
                'avg_confidence': batch.avg_confidence,
                'avg_depth_cm': batch.avg_depth_cm,
                'created_at': batch.created_at.isoformat(),
            },
            'results': batch.results_json.get('results', [])
        })
    except BatchUploadResult.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Batch not found'}, status=404)
