from django.db import models

# Create your models here.

class Team(models.Model):
    ID = models.IntegerField(primary_key=True)  # Auto-incrementing primary key
    NAME = models.CharField(max_length=150)
    EMAIL = models.EmailField(unique=True)
    PHONE = models.CharField(max_length=15, blank=True, null=True)
    PASSWORD = models.CharField(max_length=128)    
