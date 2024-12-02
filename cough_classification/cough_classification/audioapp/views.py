import librosa
from librosa import load
import numpy as np
import torch
import tempfile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
from .cnn_model import SimpleCNN 
from .models import AudioSample
from mimetypes import guess_type
import torch.nn.functional as F
import wave
#from django.shortcuts import get_object_or_404
#from django.http import HttpResponseForbidden

# Define the path to the model weights
model_path = os.path.join(os.path.dirname(__file__), "model.pth")

# Load the model
model = SimpleCNN(num_classes=3)  # Initialize the model
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))  # Load weights
model.eval()  # Set the model to evaluation mode

def classify_audio(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')

        # Validate file 
        if not audio_file:
            return JsonResponse({'error': 'No audio file provided.'}, status=400)
        
         # Validate file MIME type
        mime_type, _ = guess_type(audio_file.name)
        if mime_type != "audio/wav":
            return JsonResponse({"error": "Unsupported file type. Only WAV files are allowed."}, status=400)

        # Save audio file to a temporary location for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        with wave.open(temp_file_path, 'rb') as audio_file:
            audio_params = audio_file.getparams()
            audio_frames = audio_file.readframes(audio_params.nframes)
        audio = audio.set_frame_rate(22050).set_channels(1)
        temp_file_path = temp_file_path.replace(".wav", "_normalized.wav")
        audio.export(temp_file_path, format="wav")

        # Debug: Check if the file exists and is accessible
        print(f"Temporary file path: {temp_file_path}")
        print(f"File exists: {os.path.exists(temp_file_path)}")

        try:
            # Load the audio file using librosa
            y, sr = load(temp_file_path, sr=22050)
            print(f"Audio loaded successfully with librosa: {y.shape}, {sr}")

            # Ensure the tensor is contiguous
            if not y.flags['C_CONTIGUOUS']:
                y = np.ascontiguousarray(y)
    
            # Convert to mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Normalize spectrogram
            mel_spec_db = mel_spec_db / np.max(np.abs(mel_spec_db))

            # Convert to tensor
            mel_spec_tensor = torch.tensor(mel_spec_db)

            # Swap the height and width of the mel-spectrogram
            mel_spec_tensor = mel_spec_tensor.unsqueeze(0)  # Add batch dimension
            mel_spec_tensor = mel_spec_tensor.unsqueeze(0)  # Add channel dimension

            # Now mel_tensor should be of shape [1, 1, 128, 189] (1 channel)
            # Expand to 3 channels by duplicating the tensor
            mel_spec_tensor = mel_spec_tensor.expand(1, 3, mel_spec_tensor.shape[2], mel_spec_tensor.shape[3])


            print(f"Adjusted mel-spectrogram tensor shape: {mel_spec_tensor.shape}")
            
            # Use the model to classify the mel spectrogram
            with torch.no_grad():
                outputs = model(mel_spec_tensor)
                _, predicted = torch.max(outputs, 1)

            # Map the predicted label to a class name
            class_names = ['COVID-19', 'Healthy', 'Symptomatic']
            classification_result = class_names[predicted.item()]

            # Save audio sample to the database after successful processing
            audio_sample = AudioSample.objects.create(user=request.user, audio_file=audio_file)

            return JsonResponse({'classification': classification_result})

        except Exception as e:
            print(f"Librosa load error: {e}")
            return JsonResponse({'error': str(e)}, status=400)

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return JsonResponse({"error": "Invalid request method."}, status=405)

from django.shortcuts import render
def record_audio_page(request):
    return render(request, 'record_audio.html')
