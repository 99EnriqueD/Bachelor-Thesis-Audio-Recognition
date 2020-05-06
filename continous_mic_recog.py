import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import os
import pyaudio

# Load segment audio classification model

model_path = r"Models/"
model_name = "audio_NN_New2020_03_24_17_27_32_acc_84.44"

# Model reconstruction from JSON file
with open(model_path + model_name + ".json", 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_path + model_name + '.h5')

# Replicate label encoder
lb = LabelEncoder()
lb.fit_transform(['Glassbreak', 'Scream',
                  'Crash', 'Other'])

# Some Utils


def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)


def predictSound(X):
    # Empherically select top_db for every sample
    clip, index = librosa.effects.trim(
        X, top_db=20, frame_length=512, hop_length=64)
    stfts = np.abs(librosa.stft(clip, n_fft=512,
                                hop_length=256, win_length=512))
    stfts = np.mean(stfts, axis=1)
    stfts = minMaxNormalize(stfts)
    result = model.predict(np.array([stfts]))
    print(result)
    predictions = [np.argmax(y) for y in result]
    if np.max(result) > 0.8:
        print(lb.inverse_transform([predictions[0]])[0])
    else:
        print("Other")


CHUNKSIZE = 22050  # fixed chunk size
RATE = 22050

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                input=True, frames_per_buffer=CHUNKSIZE)

# noise window
data = stream.read(10000)
noise_sample = np.frombuffer(data, dtype=np.float32)
print("Noise Sample")
loud_threshold = np.mean(np.abs(noise_sample)) * 10
print("Loud threshold", loud_threshold)
audio_buffer = []
near = 0

while(True):
    # Read chunk and load it into numpy array.
    data = stream.read(CHUNKSIZE)
    current_window = np.frombuffer(data, dtype=np.float32)

    # Reduce noise real-time
    current_window = nr.reduce_noise(
        audio_clip=current_window, noise_clip=noise_sample, verbose=False)

    if(audio_buffer == []):
        audio_buffer = current_window
    else:
        if(np.mean(np.abs(current_window)) < loud_threshold):
            print("Inside silence reign")
            if(near < 10):
                audio_buffer = np.concatenate((audio_buffer, current_window))
                near += 1
            else:
                predictSound(np.array(audio_buffer))
                audio_buffer = []
                near = 0
        else:
            print("Inside loud reign")
            near = 0
            audio_buffer = np.concatenate((audio_buffer, current_window))

# close stream
stream.stop_stream()
stream.close()
p.terminate()
