from django.db import models
from django.contrib.auth.models import User

class OwnerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='owner_profile')
    firstName = models.CharField(max_length=50, null=False)
    lastName = models.CharField(max_length=50, null=False)
    emailId = models.EmailField(max_length=255, unique=True, null=False)
    password = models.CharField(max_length=255, null=False)
    contactNumber = models.CharField(max_length=10, unique=True, null=False)
    idProof = models.URLField(max_length=500, blank=True, null=True)
    verified = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.firstName} {self.lastName} ({self.emailId})"

class ParkingLot(models.Model):
    name = models.CharField(max_length=100, blank=True, null=True)
    location = models.CharField(max_length=255)
    total_spaces = models.PositiveIntegerField(default=0)
    available_spaces = models.PositiveIntegerField(default=0)
    registered_by = models.ForeignKey(User, on_delete=models.CASCADE)
    confirmed = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.location} ({'Confirmed' if self.confirmed else 'Pending'})"
