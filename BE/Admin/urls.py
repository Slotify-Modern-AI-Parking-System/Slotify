from django.urls import path
from . import views


urlpatterns = [
    path('loginAdmin/', views.login_admin, name='loginAdmin'),
    path('adminLoginPage/', views.adminLogin, name="AdminLoginPage"),
    path('adminDashboardPage/', views.adminDashboard, name="AdminDashboardPage"),
    path('api/run-script/', views.run_python_script, name='run_python_script'),

]