from django.urls import path
from . import views
from slotifyBE import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('userRegister/', views.userRegister, name='userRegister'),
    path('userSignIn/', views.userSignIn, name='userSignIn'),
    path('registerOwner/', views.register_owner, name='registerOwner'),
    path('registerParking/', views.parking_register_page, name='registerParkingPage'),
    path("getParkingLots/", views.get_parking_lots, name="get_parking_lots"),
    path("ownerDashboard/", views.get_owner_dashboard, name="get_owner_dashboard"),
    path('loginOwner/', views.login_owner, name='loginOwner'),
    path('options/', views.options_page, name='optionsPage'),
    path('logout/', views.logout_view, name='logout'),
    path('submitParking/', views.register_parking_lot, name='submit_parking'),
    path('nearbyParking/', views.nearby_parking, name='nearby_parking'),
]