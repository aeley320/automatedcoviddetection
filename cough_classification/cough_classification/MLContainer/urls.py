from django.urls import path
from . import views

app_name = 'MLContainer'

urlpatterns = [
    # URL for classifying the uploaded or recorded audio
    path('record_audio/', views.record_audio, name='record_audio'),
]
