from django.urls import path
from . import views

urlpatterns = [
    path('userRegister/', views.userRegister, name='userRegister'),
    path('registerOwner/', views.register_owner, name='register_owner'),
]
