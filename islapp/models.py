from django.db import models

# Create your models here.
from django.db import models
from django.utils.timezone import now

class LoginDetails(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_name = models.CharField(max_length=255, unique=True)
    password = models.CharField(max_length=255)  # Consider using Django's authentication system
    last_login = models.DateTimeField(default=now)
    contact_no = models.CharField(max_length=15)
    email = models.EmailField(unique=True)
    STATUS_CHOICES = [
        ('Active', 'Active'),
        ('Inactive', 'Inactive'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Active')
    def __str__(self):
        return self.user_name
    

# models.py
from django.db import models

class GestureVideo(models.Model):
    serial_no = models.AutoField(primary_key=True)
    gesture = models.CharField(max_length=100)
    gesture_CHOICES = [
        ('Electronics', 'Electronics'),
        ('Clothes', 'Clothes'),
        ('Seasons', 'Seasons'),
    ]
    gesture_type = models.CharField(max_length=20, choices=gesture_CHOICES, default='Electronics')
    video_file = models.FileField(upload_to='gesture_videos/')
    STATUS_CHOICES = [
        ('Active', 'Active'),
        ('Inactive', 'Inactive'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Active')
    def __str__(self):
        return f"{self.serial_no}: {self.gesture} ({self.gesture_type})"




class KnowledgeBase(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()


from django.db import models
from django.utils import timezone

class LoginAdminDetails(models.Model):
    username = models.CharField(max_length=150, unique=True)  # Unique username
    password = models.CharField(max_length=255)  # Password field
    status = models.CharField(max_length=10, choices=[('active', 'Active'), ('inactive', 'Inactive')], default='active')  # Active or Inactive status
    last_login = models.DateTimeField(default=timezone.now)  # Last login timestamp

    def __str__(self):
        return self.username
