from django.urls import include, path
from . import views

from django.conf.urls.static import static
from django.conf import settings

app_name = 'main'
urlpatterns = [
    path('', views.home, name='home'),
    path('project/', views.project, name='project'),
    path('research/', views.research, name='research'),
    path('record_audio/', views.record_audio, name='record_audio'),
    path('mlcontainer/', include('MLContainer.urls')),
    path('static/<path:file_path>/', views.serve_file, name='open_file'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
