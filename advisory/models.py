from django.db import models
from django.utils import timezone


class Product(models.Model):
    name = models.TextField()
    brand = models.CharField(max_length=200)
    text = models.TextField()

    # def publish(self):
    #     self.published_date = timezone.now()
    #     self.save()

    def __str__(self):
        return self.name