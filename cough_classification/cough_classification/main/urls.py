from django.urls import path, include
from . import views
from django.views.static import serve
from django.conf import settings

app_name = 'main'
urlpatterns = [
   
    path('', views.home, name='home'),
    path('project/', views.project, name='project'),
    path('record_audio/', views.record_audio, name='record_audio'),
    path('audioapp/', include('audioapp.urls')),
]
