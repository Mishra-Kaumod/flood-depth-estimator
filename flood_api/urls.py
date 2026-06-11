# flood_api/urls.py
from django.urls import path
from . import views

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
]
