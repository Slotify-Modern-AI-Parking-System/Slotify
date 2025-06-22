from django.urls import path
from . import views


urlpatterns = [
    # Main dashboard
    path('entry/', views.index, name='index'),
    path('welcome/', views.welcome,name="WelcomePage"),
    path('parking-lot/login/', views.parking_lot_login, name="ParkingLotLogin"),
    path('parking-lot/logout/', views.parking_lot_logout, name="ParkingLotLogout"),
    path('dashboard/', views.dashboard, name="DashboardPage"),
    path('destination/',views.destination,name="DestinationPage"),
    path('api/license-plate/', views.get_license_plate, name='get_license_plate'),
    path('api/assign-slot/', views.assign_nearest_parking_slot, name='assign_parking_slot'),

]