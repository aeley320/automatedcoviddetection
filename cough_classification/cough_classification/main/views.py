from django.shortcuts import render

# Create your views here.

def project(request):
    return render(request, 'main/project.html')


def home(request):
    return render(request, 'main/home.html')

def record_audio(request):
    return render(request, 'main/record_audio.html')