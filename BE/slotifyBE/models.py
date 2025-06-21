from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class OwnerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='owner_profile')
    firstName = models.CharField(max_length=50, null=False)
    lastName = models.CharField(max_length=50, null=False)
    emailId = models.EmailField(max_length=255, unique=True, null=False)
    password = models.CharField(max_length=255, null=False)
    contactNumber = models.CharField(max_length=10, unique=True, null=False)
    idProof = models.URLField(max_length=500, blank=True, null=True)
    verified = models.BooleanField(default=False)
    active = models.BooleanField(default=True)
    
    # New fields
    role = models.CharField(max_length=50, default="Owner")
    created_at = models.DateTimeField(default=timezone.now, null=True)
    updated_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.firstName} {self.lastName} ({self.emailId})"

class ParkingLot(models.Model):
    name = models.CharField(max_length=100, blank=True, null=True)
    location = models.CharField(max_length=255)
    total_spaces = models.PositiveIntegerField(default=0)
    available_spaces = models.PositiveIntegerField(default=0)
    registered_by = models.ForeignKey(User, on_delete=models.CASCADE)
    confirmed = models.BooleanField(default=False)
    username = models.CharField(max_length=150, unique=True, null = True, blank = True)
    password = models.CharField(max_length=128 , null = True, blank = True)


    def __str__(self):
        return f"{self.location} ({'Confirmed' if self.confirmed else 'Pending'})"


class ParkingLotCoordinate(models.Model):
    lotId = models.ForeignKey('ParkingLot', on_delete=models.CASCADE, related_name='coordinates')
    x_coordinate = models.FloatField()
    y_coordinate = models.FloatField()
    entry_x = models.FloatField()
    entry_y = models.FloatField()

    # New boolean fields
    is_regular = models.BooleanField(default=False)
    is_accessible = models.BooleanField(default=False)
    is_reservation = models.BooleanField(default=False)

    def __str__(self):
        return (
            f"Lot {self.lotId.id} - Point ({self.x_coordinate}, {self.y_coordinate}) | "
            f"Entry ({self.entry_x}, {self.entry_y})"
        )
