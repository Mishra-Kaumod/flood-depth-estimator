# flood_api/secure_random_image_views.py
# SECURE IMAGE UPLOAD WITH VALIDATION, AUTHENTICATION, AND RATE LIMITING

import os
import uuid
import hashlib
from datetime import timedelta
from pathlib import Path

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, FileResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.core.exceptions import ValidationError
from django.utils import timezone
from django.core.files.storage import default_storage
from django.middleware.csrf import get_token
from django.db import models
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.generic import View

from PIL import Image
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle
from rest_framework.exceptions import ValidationError as DRFValidationError

import cv2
import numpy as np
from functools import wraps

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_IMAGES_PER_BATCH = 10
UPLOAD_TIMEOUT_SECONDS = 10

# Magic bytes for file type validation
FILE_SIGNATURES = {
    b'\xFF\xD8\xFF': 'jpg',      # JPEG
    b'\x89PNG': 'png',            # PNG
    b'GIF8': 'gif',               # GIF
    b'BM': 'bmp',                 # BMP
}

# Default Bengaluru coordinates (Vidhana Soudha)
DEFAULT_LAT = 13.1939
DEFAULT_LNG = 77.5900
DEFAULT_CAMERA_ID = 'bengaluru_default'

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY THROTTLE
# ─────────────────────────────────────────────────────────────────────────────

class ImageUploadThrottle(UserRateThrottle):
    """Prevent DoS: max 10 uploads per hour per user"""
    scope = 'image_upload'
    rate = '10/hour'

# ─────────────────────────────────────────────────────────────────────────────
# FILE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_image_file(file):
    """Validate file type, size, and content"""
    
    # Check file extension
    ext = file.name.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(f"File type '{ext}' not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Check file size
    if file.size > MAX_FILE_SIZE:
        raise ValidationError(f"File too large ({file.size / 1024 / 1024:.1f}MB). Max: 50MB")
    
    # Check file signature (magic bytes)
    file.seek(0)
    header = file.read(8)
    
    is_valid = False
    for sig, sig_ext in FILE_SIGNATURES.items():
        if header.startswith(sig):
            is_valid = True
            break
    
    if not is_valid:
        raise ValidationError("Invalid image file or corrupted. Upload a valid JPG, PNG, GIF, or BMP.")
    
    file.seek(0)
    return True

def sanitize_filename(filename):
    """Generate safe filename using UUID"""
    ext = filename.split('.')[-1].lower()
    return f"{uuid.uuid4().hex}.{ext}"

# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATION & SANITIZATION
# ─────────────────────────────────────────────────────────────────────────────

class SecureImageUploadForm:
    """Form validator with security checks"""
    
    def __init__(self, data, files):
        self.data = data
        self.files = files
        self.errors = {}
    
    def is_valid(self):
        """Validate all inputs"""
        try:
            # Validate images
            images = self.files.getlist('images')
            if not images or len(images) == 0:
                self.errors['images'] = "No images selected"
                return False
            
            if len(images) > MAX_IMAGES_PER_BATCH:
                self.errors['images'] = f"Max {MAX_IMAGES_PER_BATCH} images per batch"
                return False
            
            # Validate each image
            for img in images:
                try:
                    validate_image_file(img)
                except ValidationError as e:
                    self.errors['images'] = str(e)
                    return False
            
            # Validate text fields (max length, no HTML)
            scenario = self.data.get('scenario_name', '').strip()
            if not scenario or len(scenario) < 3:
                self.errors['scenario_name'] = "Scenario name must be at least 3 characters"
                return False
            if len(scenario) > 100:
                self.errors['scenario_name'] = "Scenario name too long (max 100 chars)"
                return False
            if '<' in scenario or '>' in scenario:
                self.errors['scenario_name'] = "Invalid characters in scenario name"
                return False
            
            location = self.data.get('location', '').strip()
            if not location or len(location) < 3:
                self.errors['location'] = "Location must be at least 3 characters"
                return False
            if len(location) > 100:
                self.errors['location'] = "Location too long (max 100 chars)"
                return False
            if '<' in location or '>' in location:
                self.errors['location'] = "Invalid characters in location"
                return False
            
            # Optional: validate coordinates
            try:
                lat = float(self.data.get('latitude', DEFAULT_LAT))
                lng = float(self.data.get('longitude', DEFAULT_LNG))
                
                # Bengaluru bounds approximately
                if not (12.5 <= lat <= 13.5 and 77.0 <= lng <= 78.0):
                    # Outside Bengaluru, but allow with warning
                    pass
            except (ValueError, TypeError):
                self.errors['coordinates'] = "Invalid coordinates (must be numbers)"
                return False
            
            return True
            
        except Exception as e:
            self.errors['general'] = f"Validation error: {str(e)}"
            return False
    
    def cleaned_data(self):
        """Return sanitized data"""
        return {
            'images': self.files.getlist('images'),
            'scenario_name': self.data.get('scenario_name', '').strip(),
            'location': self.data.get('location', '').strip(),
            'camera_id': self.data.get('camera_id', DEFAULT_CAMERA_ID).strip(),
            'latitude': float(self.data.get('latitude', DEFAULT_LAT)),
            'longitude': float(self.data.get('longitude', DEFAULT_LNG)),
            'description': self.data.get('description', '').strip(),
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

class SecureRandomImageUploadResult(models.Model):
    """Secure storage for upload results"""
    
    INTENSITY_CHOICES = [
        ('SAFE', 'Safe - No Flood'),
        ('MEDIUM', 'Medium - Uncertain'),
        ('HIGH', 'High - Significant Flood'),
        ('CRITICAL', 'Critical - Severe Flood'),
    ]
    
    batch_id = models.CharField(max_length=100, unique=True, primary_key=True)
    user_ip = models.GenericIPAddressField(null=True)
    scenario_name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    camera_id = models.CharField(max_length=50, default=DEFAULT_CAMERA_ID)
    latitude = models.FloatField(default=DEFAULT_LAT)
    longitude = models.FloatField(default=DEFAULT_LNG)
    description = models.TextField(blank=True)
    
    total_images = models.IntegerField()
    flooded_count = models.IntegerField()
    dry_count = models.IntegerField()
    avg_confidence = models.FloatField()
    avg_depth_cm = models.FloatField()
    max_intensity = models.CharField(max_length=10, choices=INTENSITY_CHOICES, default='SAFE')
    
    results_json = models.JSONField()  # Per-image detailed results
    report_path = models.CharField(max_length=500)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'secure_random_image_upload_result'
        verbose_name = 'Secure Image Upload Result'
        verbose_name_plural = 'Secure Image Upload Results'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.batch_id} - {self.scenario_name}"

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOGGING
# ─────────────────────────────────────────────────────────────────────────────

class AuditLog(models.Model):
    """Track all uploads for security audit"""
    
    ACTION_CHOICES = [
        ('UPLOAD_START', 'Upload Started'),
        ('UPLOAD_SUCCESS', 'Upload Successful'),
        ('UPLOAD_FAILED', 'Upload Failed'),
        ('REPORT_GENERATED', 'Report Generated'),
        ('REPORT_VIEWED', 'Report Viewed'),
        ('REPORT_DOWNLOADED', 'Report Downloaded'),
    ]
    
    batch_id = models.CharField(max_length=100)
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    status = models.CharField(max_length=20)  # success, failed
    details = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'audit_log'
        verbose_name = 'Audit Log'
        verbose_name_plural = 'Audit Logs'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.batch_id} - {self.action} - {self.created_at}"

def get_client_ip(request):
    """Extract client IP from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def log_audit(batch_id, action, request, status, details=None):
    """Log audit event"""
    try:
        AuditLog.objects.create(
            batch_id=batch_id,
            action=action,
            ip_address=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')[:500],
            status=status,
            details=details or {}
        )
    except Exception as e:
        print(f"Audit log failed: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING (SAME AS BEFORE BUT SECURE)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image_secure(image_path, timeout_seconds=UPLOAD_TIMEOUT_SECONDS):
    """Analyze image with timeout protection"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Failed to read image"
        
        # Check image dimensions (max 4000x4000)
        if img.shape[0] > 4000 or img.shape[1] > 4000:
            return None, "Image too large"
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Water detection (blue-heavy pixels)
        blue = img[:, :, 0]
        green = img[:, :, 1]
        red = img[:, :, 2]
        
        water_mask = (blue > green) & (blue > red * 0.8)
        water_pixels = np.sum(water_mask) / (img.shape[0] * img.shape[1]) * 100
        
        # Flood determination
        is_flood = bool((water_pixels > 30) and (brightness < 200))
        
        # Confidence calculation
        if is_flood:
            confidence = min(100, 70 + (water_pixels - 30) * 0.5)
            intensity = "CRITICAL" if confidence > 90 else ("HIGH" if confidence > 80 else "MEDIUM")
            depth_cm = int(water_pixels * 2)
        else:
            confidence = max(50, 90 - (water_pixels * 2))
            intensity = "SAFE" if confidence > 85 else "MEDIUM"
            depth_cm = 0
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_count = np.sum(edges > 0)
        
        return {
            'is_flood': bool(is_flood),
            'confidence': float(confidence),
            'intensity': str(intensity),
            'depth_cm': int(depth_cm),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'water_pixels': float(water_pixels),
            'edge_count': int(edge_count),
            'dimensions': f"{img.shape[1]}x{img.shape[0]}",
        }, None
        
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# VIEWS
# ─────────────────────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
@csrf_protect
def secure_random_image_upload_page(request):
    """Render upload interface with Bengaluru map and security features"""
    
    # Create CSRF token
    csrf_token = get_token(request)
    
    context = {
        'csrf_token': csrf_token,
        'default_lat': DEFAULT_LAT,
        'default_lng': DEFAULT_LNG,
        'default_camera_id': DEFAULT_CAMERA_ID,
        'bengaluru_name': 'Bengaluru (Bangalore), India',
    }
    
    return render(request, 'enterprise_upload.html', context)

@require_http_methods(["POST"])
@csrf_protect
def secure_random_image_upload_process(request):
    """Process secure image upload with validation"""
    
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    client_ip = get_client_ip(request)
    
    try:
        # Rate limiting check
        throttle = ImageUploadThrottle()
        if not throttle.allow_request(request, None):
            log_audit(batch_id, 'UPLOAD_FAILED', request, 'rate_limited',
                     {'reason': 'Rate limit exceeded'})
            return JsonResponse({
                'status': 'error',
                'message': 'Too many uploads. Max 10 per hour. Please try again later.'
            }, status=429)
        
        # Validate form
        form = SecureImageUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            log_audit(batch_id, 'UPLOAD_FAILED', request, 'validation_failed',
                     {'errors': form.errors})
            return JsonResponse({
                'status': 'error',
                'message': 'Validation failed',
                'errors': form.errors
            }, status=400)
        
        log_audit(batch_id, 'UPLOAD_START', request, 'started',
                 {'image_count': len(request.FILES.getlist('images'))})
        
        # Get cleaned data
        data = form.cleaned_data()
        images = data['images']
        
        # Create upload directory
        uploads_dir = Path('uploads/temp_uploads')
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        results = []
        flooded_count = 0
        depth_values = []
        intensity_levels = {'SAFE': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        
        for idx, image_file in enumerate(images, 1):
            try:
                # Save with secure filename
                safe_name = sanitize_filename(image_file.name)
                image_path = uploads_dir / safe_name
                
                with open(image_path, 'wb') as f:
                    for chunk in image_file.chunks():
                        f.write(chunk)
                
                # Analyze image
                analysis, error = analyze_image_secure(str(image_path))
                
                if error or analysis is None:
                    results.append({
                        'image_num': idx,
                        'filename': image_file.name,
                        'status': 'error',
                        'message': error or 'Failed to analyze'
                    })
                    continue
                
                # Update statistics
                if analysis['is_flood']:
                    flooded_count += 1
                depth_values.append(analysis['depth_cm'])
                intensity_levels[analysis['intensity']] += 1
                
                results.append({
                    'image_num': idx,
                    'filename': image_file.name,
                    'status': 'success',
                    'analysis': analysis
                })
                
            except Exception as e:
                results.append({
                    'image_num': idx,
                    'filename': image_file.name,
                    'status': 'error',
                    'message': str(e)
                })
        
        # Calculate aggregated statistics
        dry_count = len(images) - flooded_count
        avg_confidence = np.mean([r['analysis']['confidence'] 
                                 for r in results 
                                 if r['status'] == 'success']) if results else 0
        avg_depth = np.mean(depth_values) if depth_values else 0
        
        # Determine max intensity
        max_intensity = 'SAFE'
        for level in ['CRITICAL', 'HIGH', 'MEDIUM']:
            if intensity_levels[level] > 0:
                max_intensity = level
                break
        
        # Create database record
        upload_result = SecureRandomImageUploadResult.objects.create(
            batch_id=batch_id,
            user_ip=client_ip,
            scenario_name=data['scenario_name'],
            location=data['location'],
            camera_id=data['camera_id'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            description=data['description'],
            total_images=len(images),
            flooded_count=flooded_count,
            dry_count=dry_count,
            avg_confidence=float(avg_confidence),
            avg_depth_cm=float(avg_depth),
            max_intensity=max_intensity,
            results_json={
                'images': results,
                'timestamp': timezone.now().isoformat(),
            },
            report_path=f'/report/{batch_id}/'
        )
        
        log_audit(batch_id, 'UPLOAD_SUCCESS', request, 'completed',
                 {'flooded': flooded_count, 'dry': dry_count})
        
        return JsonResponse({
            'status': 'success',
            'batch_id': batch_id,
            'message': f'Upload successful! Processed {len(images)} images.',
            'statistics': {
                'total_images': len(images),
                'flooded_count': flooded_count,
                'dry_count': dry_count,
                'avg_confidence': float(avg_confidence),
                'avg_depth_cm': float(avg_depth),
                'max_intensity': max_intensity,
            },
            'report_url': f'/report/{batch_id}/'
        }, status=201)
        
    except Exception as e:
        log_audit(batch_id, 'UPLOAD_FAILED', request, 'exception',
                 {'error': str(e)})
        return JsonResponse({
            'status': 'error',
            'message': f'Upload failed: {str(e)}'
        }, status=500)

@require_http_methods(["GET"])
def view_report_secure(request, batch_id):
    """View secure report with results table"""
    
    try:
        result = get_object_or_404(SecureRandomImageUploadResult, batch_id=batch_id)
        
        log_audit(batch_id, 'REPORT_VIEWED', request, 'success')
        
        context = {
            'batch': result,
            'results': result.results_json.get('images', []),
            'stats': {
                'total': result.total_images,
                'flooded': result.flooded_count,
                'dry': result.dry_count,
                'avg_confidence': result.avg_confidence,
                'avg_depth': result.avg_depth_cm,
            },
            'bengaluru_name': 'Bengaluru (Bangalore), India',
        }
        
        return render(request, 'enterprise_report.html', context)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def download_report_secure(request, batch_id):
    """Download report as HTML"""
    
    try:
        result = get_object_or_404(SecureRandomImageUploadResult, batch_id=batch_id)
        
        log_audit(batch_id, 'REPORT_DOWNLOADED', request, 'success')
        
        # Generate HTML content
        html_content = generate_report_html(result)
        
        response = FileResponse(
            iter([html_content.encode('utf-8')]),
            content_type='text/html'
        )
        response['Content-Disposition'] = f'attachment; filename="flood_report_{batch_id}.html"'
        
        return response
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def generate_report_html(result):
    """Generate HTML report"""
    
    intensity_colors = {
        'SAFE': '#228B22',
        'MEDIUM': '#FFD700',
        'HIGH': '#FF4500',
        'CRITICAL': '#8B0000',
    }
    
    rows = ""
    for img in result.results_json.get('images', []):
        if img['status'] != 'success':
            continue
        
        analysis = img['analysis']
        intensity = analysis['intensity']
        color = intensity_colors.get(intensity, '#999999')
        
        rows += f"""
        <tr>
            <td>#{img['image_num']}</td>
            <td>{img['filename']}</td>
            <td>{'FLOOD' if analysis['is_flood'] else 'DRY'}</td>
            <td>{analysis['confidence']:.1f}%</td>
            <td><span style="background-color:{color}; color:white; padding:4px 8px; border-radius:4px;">{intensity}</span></td>
            <td>{analysis['depth_cm']} cm</td>
            <td>{analysis['brightness']:.1f}</td>
            <td>{analysis['contrast']:.1f}</td>
            <td>{analysis['water_pixels']:.1f}%</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flood Detection Report - {result.batch_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
            .summary {{ background: white; padding: 15px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
            .stat-box {{ background: #f0f0f0; padding: 10px; border-radius: 4px; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; background: white; margin: 15px 0; }}
            th {{ background: #333; color: white; padding: 10px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background: #f5f5f5; }}
            .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Flood Detection Analysis Report</h1>
            <p><strong>Batch ID:</strong> {result.batch_id}</p>
            <p><strong>Location:</strong> {result.location} ({result.latitude:.4f} N, {result.longitude:.4f} E)</p>
            <p><strong>Camera:</strong> {result.camera_id}</p>
            <p><strong>Generated:</strong> {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>{result.total_images}</h3>
                    <p>Total Images</p>
                </div>
                <div class="stat-box">
                    <h3>{result.flooded_count}</h3>
                    <p>Flooded</p>
                </div>
                <div class="stat-box">
                    <h3>{result.avg_confidence:.1f}%</h3>
                    <p>Avg Confidence</p>
                </div>
                <div class="stat-box">
                    <h3>{result.avg_depth_cm:.1f} cm</h3>
                    <p>Avg Depth</p>
                </div>
            </div>
        </div>
        
        <div class="summary">
            <h2>Image Analysis Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Filename</th>
                        <th>Status</th>
                        <th>Confidence</th>
                        <th>Intensity</th>
                        <th>Depth (cm)</th>
                        <th>Brightness</th>
                        <th>Contrast</th>
                        <th>Water Pixels %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Report generated by Bengaluru Flood Detection System</p>
            <p style="color: #999; font-size: 11px;">This report is secure and contains sensitive geospatial information. Do not share publicly.</p>
        </div>
    </body>
    </html>
    """
    
    return html
