from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    # Using custom form creation
    # path('register/', views.register_view, name="register"),
    path('register/', views.register_view, name="register"),
    path('login/', views.login_view, name="login"),
    path('logout/', views.logout_view, name="logout"),
    # path('upload-audio/', views.upload_audio, name='upload_audio'),
]



