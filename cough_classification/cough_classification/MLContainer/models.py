from django.db import models
from django.conf import settings

class AudioSample(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="audio_samples")
    audio_file = models.FileField(upload_to="audio_samples/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"AudioSample({self.user.username}, {self.uploaded_at})"
