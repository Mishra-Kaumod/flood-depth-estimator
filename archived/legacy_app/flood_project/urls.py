# flood_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.url_pattern if hasattr(admin, 'url_pattern') else admin.site.urls),
    path('', include('flood_api.urls')),
]
