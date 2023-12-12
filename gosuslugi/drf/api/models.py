from django.db import models

class User(models.Model):
    name = models.CharField(max_length=255, null=False)
    age = models.PositiveIntegerField(null=False)