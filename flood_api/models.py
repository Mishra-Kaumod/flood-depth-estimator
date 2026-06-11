# flood_api/models.py
from django.db import models
from django.contrib.postgres.fields import ArrayField

class CameraLocation(models.Model):
    """
    Tracks unique camera deployment sites for temporal multi-image analysis.
    """
    camera_id = models.CharField(max_length=50, unique=True, db_index=True)
    location_name = models.CharField(max_length=255)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['camera_id']
    
    def __str__(self):
        return f"{self.camera_id} - {self.location_name}"


class FloodInundationTelemetry(models.Model):
    """
    ENHANCED: Relational schema to persist real-time sensor fusion telemetry
    with camera location tracking and reference object validation.
    """
    # 1. Temporal & Ingress Metadata
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    image_name = models.CharField(max_length=255, blank=True, null=True)
    camera = models.ForeignKey(CameraLocation, on_delete=models.PROTECT, null=True, blank=True)
    
    # 2. Engine Analytics Layers
    strategy_applied = models.CharField(max_length=150)
    surface_water_confirmed_pct = models.FloatField()
    computed_depth_cm = models.FloatField()
    system_confidence_score_pct = models.FloatField()
    
    # 3. Reference Object Tracking (HALLUCINATION PREVENTION)
    detected_reference_objects = ArrayField(models.CharField(max_length=50), default=list, blank=True)
    num_reference_objects = models.IntegerField(default=0)
    is_water_confirmed = models.BooleanField(default=False)  # Only True if multiple anchors + water detected
    
    # 4. Action Logic Gate
    safety_risk_assessment = models.CharField(max_length=150)

    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = "Flood Inundation Telemetry Records"
        indexes = [
            models.Index(fields=['camera', '-timestamp']),
            models.Index(fields=['is_water_confirmed', '-timestamp']),
        ]

    def __str__(self):
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Camera: {self.camera.camera_id if self.camera else 'Unknown'} | Depth: {self.computed_depth_cm}cm | Confirmed: {self.is_water_confirmed}"


class TemporalFloodSequence(models.Model):
    """
    Groups multiple images from same camera over 5-15 minute intervals
    for reliable depth estimation using multiple reference objects.
    """
    camera = models.ForeignKey(CameraLocation, on_delete=models.CASCADE)
    sequence_start = models.DateTimeField(db_index=True)
    sequence_end = models.DateTimeField()
    image_count = models.IntegerField(default=0)
    
    # Aggregated Results
    average_depth_cm = models.FloatField(null=True, blank=True)
    max_depth_cm = models.FloatField(null=True, blank=True)
    min_depth_cm = models.FloatField(null=True, blank=True)
    
    # Consensus Detection
    water_detected_in_images = models.IntegerField(default=0)
    detected_anchor_types = ArrayField(models.CharField(max_length=50), default=list, blank=True)
    consensus_water_present = models.BooleanField(default=False)
    confidence_score = models.FloatField(default=0.0)
    
    telemetry_records = models.ManyToManyField(FloodInundationTelemetry)
    
    class Meta:
        ordering = ['-sequence_start']
        verbose_name_plural = "Temporal Flood Sequences"
        indexes = [
            models.Index(fields=['camera', '-sequence_start']),
        ]
    
    def __str__(self):
        return f"{self.camera.camera_id} | {self.sequence_start.strftime('%Y-%m-%d %H:%M')} | Depth: {self.average_depth_cm}cm | Consensus: {self.consensus_water_present}"
