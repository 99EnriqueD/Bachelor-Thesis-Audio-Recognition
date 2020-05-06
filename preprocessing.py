import librosa
import noisereduce as nr
import os
import numpy as np

"""
Script that preprocesses wav files to npy files usable for the ML models

This script transforms the audio to frequencies using the STFT.
Meant to be used for the classificationDENSE.py and classificationCONV.py models.
"""

sound_classes = ['Glassbreak', 'Scream',
                 'Crash', 'Other', 'Watersounds']

subjects = ['s01', 's02', 's03', 's04', 's05']


def save_STFT(file, name, activity, subject):
    # read audio data
    audio_data, sample_rate = librosa.load(file)

    # noise reduction
    noisy_part = audio_data[0:25000]
    reduced_noise = nr.reduce_noise(
        audio_clip=audio_data, noise_clip=noisy_part, verbose=False)

    # trimming
    trimmed, index = librosa.effects.trim(
        reduced_noise, top_db=20, frame_length=512, hop_length=64)

    # extract frequency features
    stft = np.abs(librosa.stft(trimmed, n_fft=512,
                               hop_length=256, win_length=512))
    # save features
    np.save("STFT_features/stft_257_1/" + subject + "_" +
            name[:-4] + "_" + activity + ".npy", stft)


# Iterates over all sound_classes and subjects to perform the preprocessing on each wav file in the dataset.
for s_class in sound_classes:
    for subject in subjects:
        innerDir = subject + "/" + s_class
        for file in os.listdir("Dataset_audio/" + innerDir):
            if(file.endswith(".wav")):
                save_STFT("Dataset_audio/" + innerDir +
                          "/" + file, file, s_class, subject)
                print(subject, s_class, file)
