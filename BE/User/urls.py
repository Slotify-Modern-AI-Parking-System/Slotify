from django.urls import path
from . import views


urlpatterns = [
    # Main dashboard
    path('entry/', views.index, name='index'),
    
    # Detection control endpoints
    path('start/', views.start_detection, name='start_detection'),
    path('stop/', views.stop_detection, name='stop_detection'),
    
    # Plate detection endpoints
    path('plate-detected/', views.plate_detected, name='plate_detected'),
    path('confirm-plate/', views.confirm_plate, name='confirm_plate'),
    
    # Status endpoints
    path('status/', views.get_status, name='get_status'),
    path('stream/', views.detection_stream, name='detection_stream'),
]