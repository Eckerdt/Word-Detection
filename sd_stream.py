import sounddevice as sd
import numpy as np

import timeit

from scipy.signal import butter, lfilter
from librosa.feature import mfcc

from tflite_runtime.interpreter import Interpreter

import time


# Parameters
audio_length= 1.5
debug_time = 1
debug_acc = 0

word_threshold = 0.5
rec_duration = 0.6
stopp= 0
sample_rate = 22050

num_channels = 1
model_path = '4_layer_globalpool.tflite'

# Sliding window
window = np.zeros(int(audio_length*22050))



# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def timer(stream, duration):
    stop=time.time()+duration
    with stream:
        while time.time()< stop:
            pass

def ZScore(data):
    Mean = np.mean(data)
    Std = np.std(data)
    data_ZScore = np.array((data - Mean) / Std)
    return data_ZScore





# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):
    global stopp
    stopp=False

    # Start timing for testing
    start = timeit.default_timer()

    # Notify if errors
    if status:
        print('Error:', status)

    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)

    # Filter
    rec= butter_lowpass_filter(cutoff=10000,data=rec, fs= sample_rate, order=5)


    window_part=window[int(sample_rate*rec_duration):]
    window[:(len(window)-int(sample_rate*rec_duration))]=window_part
    window[len(window_part):]=rec



    # Compute features
    mfcc_= mfcc(y=window, sr=sample_rate, n_mfcc=20, n_fft=512)


    mfccs_=ZScore(mfcc_)

    pad_width = 1
    mfccs = np.pad(mfccs_, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, (mfccs.shape[0]), mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.around(output_data[0], 2)

    location_dict = {"Aufenthaltsraum": output_data[0], "Cafe": output_data[1], "Gruppenraum": output_data[2],
                     "Ruheraum": output_data[3], "Schlafzimmer": output_data[4], "Speisesaal": output_data[5]}

    print(location_dict)
    if np.amax(output_data) > word_threshold:
        location_list = list(location_dict)

        print("####   " + location_list[np.argmax(output_data)] + "   ####")
        stopp=True






    if debug_time:
        print("Time needed: "+ str(timeit.default_timer() - start))





#timer(stream=sd.InputStream(channels=num_channels,samplerate=sample_rate, blocksize=int(sample_rate * rec_duration), callback=sd_callback), duration=5)


with sd.InputStream(channels=num_channels,samplerate=sample_rate, blocksize=int(sample_rate * rec_duration), callback=sd_callback):
        while stopp == False:
            pass




