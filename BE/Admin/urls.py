from django.urls import path
from . import views


urlpatterns = [
    path('loginAdmin/', views.login_admin, name='loginAdmin'),
    path('adminLoginPage/', views.adminLogin, name="AdminLoginPage"),
    path('adminDashboardPage/', views.adminDashboard, name="AdminDashboardPage"),
    path('api/run-script/', views.run_python_script, name='run_python_script'),
    path('api/unconfirmed-parkinglots/', views.unconfirmed_parkinglots, name='unconfirmed_parkinglots'),
    path('upload-image/', views.upload_image, name='upload_image'),
    path('api/dashboard/counts/', views.dashboard_counts, name='dashboard_counts'),
    path('userManagement/', views.userManagement, name="UserManagementPage"),
    path("userManagementSummary/", views.user_summary_and_list, name="UserManagementSummary"),
    path("suspend_account/", views.suspend_account, name="SuspendAccount"),
    path("delete_account/", views.delete_account, name="DeleteAccount"),
    path("reactivate_account/", views.reactivate_account, name="ReactivateAccount"),
    path('reports/', views.reports, name="ReportsPage"),
    path('api/parking-stats/', views.parking_lot_stats, name='parking_lot_stats'),
    path('revenueTrends/',views.revenue_trend_chart,name="RevenuTrendChart"),
    path('reservationsTrends/',views.reservations_trends,name="ReservationsTrendsChart"),
    path('top-performing-parking-lots/', views.top_performing_parking_lots, name='top-performing-parking-lots'),
     path('reservations_financial_summary/', views.reservations_financial_summary, name='reservations_financial_summary'),
     path('recent_parking_reservations/', views.recent_parking_reservations, name='recent_parking_reservations'),
     path('detectionSummary/', views.detection_summary, name="DetectionSummaryStats"),
     path('paymentAnalytics/',views.payment_analytics, name="PaymentStats")
]

