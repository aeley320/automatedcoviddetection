from django.urls import path
from . import views

urlpatterns = [
    # URL for classifying the uploaded or recorded audio
    path('classify/', views.classify_audio, name='classify_audio'),
]
