from django.db import models
from django.utils import timezone

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
# Create your models here.
