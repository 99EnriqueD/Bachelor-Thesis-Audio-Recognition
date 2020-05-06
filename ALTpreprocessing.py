import librosa
import noisereduce as nr
import os
import numpy as np

"""
Script that preprocesses wav files to npy files usable for the ML models

This script does not use the STFT and is meant to be used for the ALTclassificationCONV.py models.
"""


def save_audionpy(file, name, activity, subject):
    # read audio data
    audio_data, sample_rate = librosa.load(file)

    # noise reduction
    noisy_part = audio_data[0:25000]
    reduced_noise = nr.reduce_noise(
        audio_clip=audio_data, noise_clip=noisy_part, verbose=False)

    # trimming
    trimmed, index = librosa.effects.trim(
        reduced_noise, top_db=20, frame_length=512, hop_length=64)

    # save features
    np.save("NONSTFT_features/timeseries/" + subject + "_" +
            name[:-4] + "_" + activity + ".npy", trimmed)


activities = ['Glassbreak', 'Scream',
              'Crash', 'Other', 'Watersounds']

subjects = ['s01', 's02', 's03', 's04', 's05']

for activity in activities:
    for subject in subjects:
        innerDir = subject + "/" + activity
        for file in os.listdir("Dataset_audio/" + innerDir):
            if(file.endswith(".wav")):
                save_audionpy("Dataset_audio/" + innerDir +
                              "/" + file, file, activity, subject)
                print(subject, activity, file)

