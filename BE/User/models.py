from django.db import models
from django.utils import timezone
from slotifyBE.models import ParkingLot, ParkingLotCoordinate

class LicensePlateDetection(models.Model):
    plate_number = models.CharField(max_length=20)
    camera_id = models.IntegerField()
    detection_time = models.DateTimeField(default=timezone.now)
    confidence_score = models.FloatField(default=0.0)
    is_confirmed = models.BooleanField(default=False)
    user_confirmed = models.BooleanField(default=False)
    session_id = models.CharField(max_length=100, unique=True)
    
    class Meta:
        ordering = ['-detection_time']
    
    def __str__(self):
        return f"{self.plate_number} - Camera {self.camera_id}"

# class ParkingReservation(models.Model):
#     name = models.CharField(max_length=100)
#     email = models.EmailField()
#     license_plate = models.CharField(max_length=20)
#     hours = models.IntegerField()
#     total_amount = models.DecimalField(max_digits=10, decimal_places=2)
#     created_at = models.DateTimeField(auto_now_add=True)
    
#     def __str__(self):
#         return f"{self.name} - {self.license_plate}"

# class Payment(models.Model):
#     reservation = models.ForeignKey(ParkingReservation, on_delete=models.CASCADE)
#     stripe_payment_intent_id = models.CharField(max_length=200)
#     amount = models.DecimalField(max_digits=10, decimal_places=2)
#     status = models.CharField(max_length=20, default='pending')
#     created_at = models.DateTimeField(auto_now_add=True)
    
#     def __str__(self):
#         return f"Payment for {self.reservation.name}"


class ParkingReservation(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    license_plate = models.CharField(max_length=20)
    location = models.CharField(max_length=255, null = True, blank = True)
    parking_lot = models.ForeignKey(ParkingLot, on_delete=models.CASCADE, null=True, blank=True)
    assigned_slot = models.ForeignKey(ParkingLotCoordinate, on_delete=models.CASCADE, null=True, blank=True)
    hours = models.IntegerField()
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.license_plate}"

class Payment(models.Model):
    reservation = models.ForeignKey(ParkingReservation, on_delete=models.CASCADE)
    stripe_payment_intent_id = models.CharField(max_length=200)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=20, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Payment for {self.reservation.name}"


# Create your models here.
