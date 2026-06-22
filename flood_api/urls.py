# flood_api/urls.py
from django.urls import path
from . import views
from . import secure_random_image_views

urlpatterns = [
    # Mapping the empty path redirects root traffic to the dashboard instantly
    path('', views.dashboard_view, name='web_dashboard'),
    path('dashboard/', views.dashboard_view, name='web_dashboard_alias'),
    
    # High-speed API endpoints
    path('api/v1/estimate/', views.high_speed_api_endpoint, name='rapid_api_gateway'),
    
    # ENHANCED: Temporal analysis endpoints
    path('api/v1/temporal/<str:camera_id>/', views.get_temporal_sequence, name='get_temporal_sequence'),
    path('api/v1/temporal/<str:camera_id>/analyze/', views.trigger_temporal_analysis, name='trigger_temporal_analysis'),
    path('api/v1/camera/<str:camera_id>/stats/', views.get_camera_stats, name='get_camera_stats'),
    path('api/v1/health/live/', views.health_live, name='health_live'),
    path('api/v1/health/ready/', views.health_ready, name='health_ready'),
    path('api/v1/ops/metrics/', views.ops_metrics_api, name='ops_metrics_api'),
    path('api/v1/ingest/batch/', views.batch_ingest_api, name='batch_ingest_api'),
    path('api/v1/telemetry/map-points/', views.telemetry_map_points_api, name='telemetry_map_points_api'),
    path('api/v1/telemetry/<int:telemetry_id>/', views.telemetry_detail_api, name='telemetry_detail_api'),
    path('api/v1/feedback/', views.feedback_submit_api, name='feedback_submit_api'),
    path('api/v1/feedback/queue/', views.feedback_queue_api, name='feedback_queue_api'),
    path('api/v1/models/', views.model_versions_api, name='model_versions_api'),
    
    # Active learning & retraining endpoints
    path('api/v1/floods/verify/', views.verify_prediction, name='verify_prediction'),
    path('api/v1/floods/feedback-summary/', views.feedback_summary_api, name='feedback_summary_api'),
    path('api/v1/ml-ops/retrain-trigger-manual/', views.retrain_trigger_manual, name='retrain_trigger_manual'),
    path('api/v1/ml-ops/model-promotion/', views.model_promotion_api, name='model_promotion_api'),
    
    # ─────────────────────────────────────────────────────────────────────────
    # SECURE RANDOM IMAGE UPLOAD SYSTEM (NEW)
    # ─────────────────────────────────────────────────────────────────────────
    path('random-upload/', secure_random_image_views.secure_random_image_upload_page, name='secure_upload_page'),
    path('api/v1/floods/random-upload-secure/', secure_random_image_views.secure_random_image_upload_process, name='secure_upload_process'),
    path('report/<str:batch_id>/', secure_random_image_views.view_report_secure, name='secure_report_view'),
    path('download-report/<str:batch_id>/', secure_random_image_views.download_report_secure, name='secure_report_download'),
]
