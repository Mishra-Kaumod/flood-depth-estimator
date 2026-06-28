from django.db import models


class CameraLocation(models.Model):
    """
    Tracks unique camera deployment sites for temporal multi-image analysis.
    """

    camera_id = models.CharField(max_length=50, unique=True, db_index=True)
    location_id = models.CharField(max_length=80, null=True, blank=True, db_index=True)
    location_name = models.CharField(max_length=255)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["camera_id"]

    def __str__(self):
        return f"{self.camera_id} - {self.location_name}"


class FloodInundationTelemetry(models.Model):
    """
    Relational schema to persist real-time sensor fusion telemetry
    with camera location tracking and reference object validation.
    """

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    image_name = models.CharField(max_length=255, blank=True, null=True)
    camera = models.ForeignKey(CameraLocation, on_delete=models.PROTECT, null=True, blank=True)

    strategy_applied = models.CharField(max_length=150)
    surface_water_confirmed_pct = models.FloatField()
    computed_depth_cm = models.FloatField()
    system_confidence_score_pct = models.FloatField()

    detected_reference_objects = models.JSONField(default=list, blank=True)
    num_reference_objects = models.IntegerField(default=0)
    is_water_confirmed = models.BooleanField(default=False)

    safety_risk_assessment = models.CharField(max_length=150)

    class Meta:
        ordering = ["-timestamp"]
        verbose_name_plural = "Flood Inundation Telemetry Records"
        indexes = [
            models.Index(fields=["camera", "-timestamp"]),
            models.Index(fields=["is_water_confirmed", "-timestamp"]),
        ]

    def __str__(self):
        camera_id = self.camera.camera_id if self.camera else "Unknown"
        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Camera: {camera_id} | Depth: {self.computed_depth_cm}cm | "
            f"Confirmed: {self.is_water_confirmed}"
        )


class TemporalFloodSequence(models.Model):
    """
    Groups multiple images from same camera over 5-15 minute intervals
    for reliable depth estimation using multiple reference objects.
    """

    camera = models.ForeignKey(CameraLocation, on_delete=models.CASCADE)
    sequence_start = models.DateTimeField(db_index=True)
    sequence_end = models.DateTimeField()
    image_count = models.IntegerField(default=0)

    average_depth_cm = models.FloatField(null=True, blank=True)
    max_depth_cm = models.FloatField(null=True, blank=True)
    min_depth_cm = models.FloatField(null=True, blank=True)

    water_detected_in_images = models.IntegerField(default=0)
    detected_anchor_types = models.JSONField(default=list, blank=True)
    consensus_water_present = models.BooleanField(default=False)
    confidence_score = models.FloatField(default=0.0)

    telemetry_records = models.ManyToManyField(FloodInundationTelemetry)

    class Meta:
        ordering = ["-sequence_start"]
        verbose_name_plural = "Temporal Flood Sequences"
        indexes = [
            models.Index(fields=["camera", "-sequence_start"]),
        ]

    def __str__(self):
        return (
            f"{self.camera.camera_id} | {self.sequence_start.strftime('%Y-%m-%d %H:%M')} | "
            f"Depth: {self.average_depth_cm}cm | Consensus: {self.consensus_water_present}"
        )


class ModelVersion(models.Model):
    model_name = models.CharField(max_length=120, db_index=True)
    version = models.CharField(max_length=64, db_index=True)
    stage = models.CharField(max_length=32, default="candidate", db_index=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        unique_together = ("model_name", "version")

    def __str__(self):
        return f"{self.model_name}:{self.version} [{self.stage}]"


class PredictionFeedback(models.Model):
    FEEDBACK_CHOICES = [
        ("accepted", "Accepted"),
        ("rejected", "Rejected"),
        ("corrected", "Corrected"),
    ]

    telemetry = models.ForeignKey(
        FloodInundationTelemetry,
        on_delete=models.CASCADE,
        related_name="feedback_entries",
    )
    feedback_type = models.CharField(max_length=16, choices=FEEDBACK_CHOICES)
    corrected_depth_cm = models.FloatField(null=True, blank=True)
    corrected_risk = models.CharField(max_length=150, blank=True)
    reviewer = models.CharField(max_length=120, blank=True)
    notes = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Feedback<{self.feedback_type}> telemetry={self.telemetry_id}"


class IngestIdempotencyKey(models.Model):
    endpoint = models.CharField(max_length=120, db_index=True)
    key = models.CharField(max_length=120, unique=True)
    response_status = models.IntegerField(default=202)
    response_payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["endpoint", "-created_at"]),
        ]

    def __str__(self):
        return f"Idempotency<{self.endpoint}:{self.key}>"


class FailedTaskEvent(models.Model):
    task_name = models.CharField(max_length=180, db_index=True)
    source_endpoint = models.CharField(max_length=120, blank=True, db_index=True)
    payload = models.JSONField(default=dict, blank=True)
    error_message = models.TextField()
    retry_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    resolved = models.BooleanField(default=False, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["resolved", "-created_at"]),
            models.Index(fields=["source_endpoint", "-created_at"]),
        ]

    def __str__(self):
        return f"FailedTask<{self.task_name}> retries={self.retry_count} resolved={self.resolved}"


class SecureRandomImageUploadResult(models.Model):
    INTENSITY_CHOICES = [
        ("SAFE", "Safe - No Flood"),
        ("MEDIUM", "Medium - Uncertain"),
        ("HIGH", "High - Significant Flood"),
        ("CRITICAL", "Critical - Severe Flood"),
    ]

    batch_id = models.CharField(max_length=100, unique=True, primary_key=True)
    user_ip = models.GenericIPAddressField(null=True)
    scenario_name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    camera_id = models.CharField(max_length=50, default="bengaluru_default")
    latitude = models.FloatField(default=13.1939)
    longitude = models.FloatField(default=77.59)
    description = models.TextField(blank=True)

    total_images = models.IntegerField()
    flooded_count = models.IntegerField()
    dry_count = models.IntegerField()
    avg_confidence = models.FloatField()
    avg_depth_cm = models.FloatField()
    max_intensity = models.CharField(max_length=10, choices=INTENSITY_CHOICES, default="SAFE")

    results_json = models.JSONField()
    report_path = models.CharField(max_length=500)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "secure_random_image_upload_result"
        verbose_name = "Secure Image Upload Result"
        verbose_name_plural = "Secure Image Upload Results"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.batch_id} - {self.scenario_name}"


class AuditLog(models.Model):
    ACTION_CHOICES = [
        ("UPLOAD_START", "Upload Started"),
        ("UPLOAD_SUCCESS", "Upload Successful"),
        ("UPLOAD_FAILED", "Upload Failed"),
        ("REPORT_GENERATED", "Report Generated"),
        ("REPORT_VIEWED", "Report Viewed"),
        ("REPORT_DOWNLOADED", "Report Downloaded"),
    ]

    batch_id = models.CharField(max_length=100)
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    status = models.CharField(max_length=20)
    details = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "audit_log"
        verbose_name = "Audit Log"
        verbose_name_plural = "Audit Logs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.batch_id} - {self.action} - {self.created_at}"
