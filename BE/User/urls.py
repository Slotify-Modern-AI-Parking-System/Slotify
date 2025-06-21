from django.urls import path
from . import views


urlpatterns = [
    # Main dashboard
    path('entry/', views.index, name='index'),
    path('welcome/', views.welcome,name="WelcomePage"),
    path('parking-lot/login/', views.parking_lot_login, name="ParkingLotLogin"),
    path('parking-lot/logout/', views.parking_lot_logout, name="ParkingLotLogout"),

]