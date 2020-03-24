import librosa
import noisereduce as nr
import os
import numpy as np


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

    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512,
                               hop_length=256, win_length=512))
    # save features
    np.save("STFT_features/stft_257_1/" + subject + "_" +
            name[:-4] + "_" + activity + ".npy", stft)

# From example repo:
# activities = ['Calling', 'Clapping', 'Drinking', 'Eating', 'Entering',
#               'Exiting', 'Falling', 'LyingDown', 'OpeningPillContainer',
#               'PickingObject', 'Reading', 'SitStill', 'Sitting', 'Sleeping',
#               'StandUp', 'Sweeping', 'UseLaptop', 'UsingPhone', 'WakeUp', 'Walking',
#               'WashingHand', 'WatchingTV', 'WaterPouring', 'Writing']


activities = ["Watersounds"]

# From example repo:
# subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09',
#             's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17']

subjects = ['s01', 's02', 's03', 's04', 's05']

for activity in activities:
    for subject in subjects:
        innerDir = subject + "/" + activity
        for file in os.listdir("Dataset_audio/" + innerDir):
            if(file.endswith(".wav")):
                save_STFT("Dataset_audio/" + innerDir +
                          "/" + file, file, activity, subject)
                print(subject, activity, file)
