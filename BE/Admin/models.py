from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class AdminProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='admin_profile')
    firstName = models.CharField(max_length=50, null=False)
    lastName = models.CharField(max_length=50, null=False)
    emailId = models.EmailField(max_length=255, unique=True, null=False)
    password = models.CharField(max_length=255, null=False)
    contactNumber = models.CharField(max_length=10, unique=True, null=False)

    def __str__(self):
        return f"{self.firstName} {self.lastName} ({self.emailId})"

