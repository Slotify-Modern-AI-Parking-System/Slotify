from django.db import models

class OwnerProfile(models.Model):
    firstName = models.CharField(max_length=50, null=False)
    lastName = models.CharField(max_length=50, null=False)
    emailId = models.EmailField(max_length=255, unique=True, null=False)
    password = models.CharField(max_length=255, null=False)
    contactNumber = models.CharField(max_length=10, unique=True, null=False)
    idProof = models.URLField(max_length=500, blank=True, null=True)
    verified = models.BooleanField(default=False)