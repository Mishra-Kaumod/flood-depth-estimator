"""
Enhanced Django view for random image upload with HTML report generation
"""

from django.shortcuts import render, redirect
from django.http import JsonResponse, FileResponse
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.core.files.storage import default_storage
from django.db import models
from django import forms

import os
import json
from pathlib import Path
from datetime import datetime
import uuid

from random_image_upload import RandomImageUploadProcessor


class RandomImageUploadForm(forms.Form):
    """Form for uploading random images"""
    images = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True, 'accept': 'image/*'}),
        label='Upload Images (1-10)',
        help_text='Drag and drop or click to select images'
    )
    scenario_name = forms.CharField(
        max_length=100,
        required=True,
        initial='Flood Detection Test',
        widget=forms.TextInput(attrs={'placeholder': 'e.g., Bengaluru Monsoon Test'})
    )
    location = forms.CharField(
        max_length=200,
        required=False,
        initial='Bengaluru, India',
        widget=forms.TextInput(attrs={'placeholder': 'e.g., Indiranagar, Whitefield'})
    )
    description = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.Textarea(attrs={
            'placeholder': 'Optional: Describe the test scenario',
            'rows': 3
        })
    )


class RandomImageUploadResult(models.Model):
    """Store random image upload results"""
    batch_id = models.CharField(max_length=50, unique=True)
    scenario_name = models.CharField(max_length=100)
    location = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    
    total_images = models.IntegerField()
    flooded_count = models.IntegerField()
    dry_count = models.IntegerField()
    
    avg_confidence = models.FloatField()
    avg_depth_cm = models.FloatField()
    
    results_json = models.JSONField()
    report_path = models.CharField(max_length=500)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.scenario_name} - {self.total_images} images"


@require_http_methods(["GET", "POST"])
def random_image_upload(request):
    """Handle random image upload and report generation"""
    
    if request.method == 'POST':
        form = RandomImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                # Get uploaded files
                uploaded_files = request.FILES.getlist('images')
                if not uploaded_files:
                    return JsonResponse({
                        'success': False,
                        'error': 'No files uploaded'
                    }, status=400)
                
                if len(uploaded_files) > 10:
                    return JsonResponse({
                        'success': False,
                        'error': 'Maximum 10 images allowed'
                    }, status=400)
                
                # Save uploaded images temporarily
                temp_dir = Path(settings.MEDIA_ROOT) / 'temp_uploads'
                temp_dir.mkdir(exist_ok=True)
                
                saved_paths = []
                for uploaded_file in uploaded_files:
                    # Generate safe filename
                    filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
                    file_path = temp_dir / filename
                    
                    # Save file
                    with open(file_path, 'wb') as f:
                        for chunk in uploaded_file.chunks():
                            f.write(chunk)
                    
                    saved_paths.append(str(file_path))
                
                # Process images
                processor = RandomImageUploadProcessor()
                results, report_path = processor.process_batch(saved_paths)
                
                # Calculate statistics
                flooded_count = sum(1 for r in results if r['is_flood'])
                dry_count = len(results) - flooded_count
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                avg_depth = sum(r['depth_cm'] for r in results) / len(results) if flooded_count > 0 else 0
                
                # Generate batch ID
                batch_id = str(uuid.uuid4())[:8]
                
                # Save to database
                upload_result = RandomImageUploadResult.objects.create(
                    batch_id=batch_id,
                    scenario_name=form.cleaned_data['scenario_name'],
                    location=form.cleaned_data['location'],
                    description=form.cleaned_data['description'],
                    total_images=len(results),
                    flooded_count=flooded_count,
                    dry_count=dry_count,
                    avg_confidence=avg_confidence,
                    avg_depth_cm=avg_depth,
                    results_json=json.dumps(results),
                    report_path=report_path
                )
                
                # Clean up temp files
                for path in saved_paths:
                    try:
                        os.remove(path)
                    except:
                        pass
                
                return JsonResponse({
                    'success': True,
                    'batch_id': batch_id,
                    'message': f'Successfully processed {len(results)} images'
                })
            
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': str(e)
                }, status=500)
        
        else:
            return JsonResponse({
                'success': False,
                'errors': form.errors
            }, status=400)
    
    else:
        form = RandomImageUploadForm()
        return render(request, 'random_image_upload.html', {'form': form})


@require_http_methods(["GET"])
def view_report(request, batch_id):
    """View generated HTML report"""
    try:
        result = RandomImageUploadResult.objects.get(batch_id=batch_id)
        report_path = Path(result.report_path)
        
        if report_path.exists():
            with open(report_path, 'r') as f:
                html_content = f.read()
            return render(request, 'report_viewer.html', {
                'html_content': html_content,
                'batch_id': batch_id,
                'scenario_name': result.scenario_name
            })
        else:
            return JsonResponse({'error': 'Report not found'}, status=404)
    
    except RandomImageUploadResult.DoesNotExist:
        return JsonResponse({'error': 'Batch not found'}, status=404)


@require_http_methods(["GET"])
def download_report(request, batch_id):
    """Download HTML report as file"""
    try:
        result = RandomImageUploadResult.objects.get(batch_id=batch_id)
        report_path = Path(result.report_path)
        
        if report_path.exists():
            return FileResponse(
                open(report_path, 'rb'),
                as_attachment=True,
                filename=f'flood_report_{batch_id}.html'
            )
        else:
            return JsonResponse({'error': 'Report not found'}, status=404)
    
    except RandomImageUploadResult.DoesNotExist:
        return JsonResponse({'error': 'Batch not found'}, status=404)


@require_http_methods(["GET"])
def list_uploads(request):
    """List all uploaded batches"""
    uploads = RandomImageUploadResult.objects.all()[:50]
    
    results = []
    for upload in uploads:
        results.append({
            'batch_id': upload.batch_id,
            'scenario_name': upload.scenario_name,
            'location': upload.location,
            'total_images': upload.total_images,
            'flooded_count': upload.flooded_count,
            'avg_confidence': round(upload.avg_confidence, 1),
            'created_at': upload.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return JsonResponse({
        'total': len(uploads),
        'results': results
    })


@require_http_methods(["GET"])
def get_batch_results(request, batch_id):
    """Get results for a specific batch"""
    try:
        result = RandomImageUploadResult.objects.get(batch_id=batch_id)
        
        return JsonResponse({
            'batch_id': batch_id,
            'scenario_name': result.scenario_name,
            'location': result.location,
            'description': result.description,
            'total_images': result.total_images,
            'flooded_count': result.flooded_count,
            'dry_count': result.dry_count,
            'avg_confidence': result.avg_confidence,
            'avg_depth_cm': result.avg_depth_cm,
            'results': json.loads(result.results_json),
            'created_at': result.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except RandomImageUploadResult.DoesNotExist:
        return JsonResponse({'error': 'Batch not found'}, status=404)
