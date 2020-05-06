import numpy as np
import os
import pyaudio
import time


CHUNKSIZE = 22050  # fixed chunk size
RATE = 22050

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                input=True, frames_per_buffer=CHUNKSIZE)

# noise window used to calibrate the loud threshold
data = stream.read(10000)
noise_sample = np.frombuffer(data, dtype=np.float32)
print("Noise Sample")

# how much louder than normal street noise counts as loud? (law is subjective)
loud_factor = 20
loud_threshold = np.mean(np.abs(noise_sample)) * loud_factor
print("Loud threshold", loud_threshold)

audio_buffer = []
count = 0
count_threshold = 10,
start_timer = time.time()
time_out = 60*1000  # = one minute

while(True):
    # Read chunk and load it into numpy array.
    data = stream.read(CHUNKSIZE)
    current_window = np.frombuffer(data, dtype=np.float32)

    if(audio_buffer == []):
        audio_buffer = current_window
    else:
        if(np.mean(np.abs(current_window)) < loud_threshold):
            print("Inside silence reign")
        else:
            print("Inside loud reign")
            # a loud noise has been detected
            count += 1
            elapsed_time = start_timer - time.time()

            if (elapsed_time > time_out):

                if (count >= count_threshold):
                    # SEND A LOUD NOISE REPORT HERE
                    print("LOUD NOISE")
                    count = 0

                # reset vars
                count = 0
                start_timer = time.time()

            audio_buffer = np.concatenate((audio_buffer, current_window))

# close stream
stream.stop_stream()
stream.close()
p.terminate()
