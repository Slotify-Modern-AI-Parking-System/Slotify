from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('userRegister/', views.userRegister, name='userRegister'),
    path('registerOwner/', views.register_owner, name='registerOwner'),
    path("registerParking/", views.register_parking_lot, name="register_parking_lot"),
    path("getParkingLots/", views.get_parking_lots, name="get_parking_lots"),
    path("ownerDashboard/", views.get_owner_dashboard, name="get_owner_dashboard"),
]
