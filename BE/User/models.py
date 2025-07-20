from django.db import models
from django.utils import timezone
from slotifyBE.models import ParkingLot, ParkingLotCoordinate

class LicensePlateDetection(models.Model):
    plate_number = models.CharField(max_length=20)
    location = models.ForeignKey(ParkingLot, on_delete=models.CASCADE, null=True, blank=True)
    entry_time = models.DateTimeField(null=True, blank=True)
    exit_time = models.DateTimeField(null=True, blank=True)
    detection_similarity_percentage = models.FloatField(null=True, blank=True, default=85)

    def __str__(self):
        return f"{self.plate_number} - {self.location.location}"
     
class Customer(models.Model):
    full_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255, blank = True, null=False)
    phone = models.CharField(max_length=15, null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.full_name} ({self.license_plate})"
    

class ParkingReservation(models.Model):
    # models.py
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, null=True, blank=True, related_name='reservations')
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


