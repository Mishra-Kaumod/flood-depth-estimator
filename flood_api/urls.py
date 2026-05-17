# flood_api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Main manual upload frontend interface
    path('', views.upload_ui, name='upload_ui'), 
    
    # Unified programmatic API endpoint for single images, batch images, or videos
    path('analyze/', views.analyze_media, name='analyze_media'), 
]