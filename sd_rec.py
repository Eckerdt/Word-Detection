from sounddevice import rec, wait


###
sample_rate=22050
duration= 6


print("Starting recording..")
rec =rec(int(duration*sample_rate),samplerate=sample_rate, channels=1, blocking=False)
wait()
print("Recording finished..")


###
rec_duration=1.5
sec_slide=0.1
word_threshold=0.5


from librosa.feature import mfcc
import numpy as np
from scipy.signal import butter, lfilter
from tflite_runtime.interpreter import Interpreter


def ZScore(data):
    Mean = np.mean(data)
    Std = np.std(data)
    data_ZScore = np.array((data - Mean) / Std)
    return data_ZScore

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y





#remove 2nd dim
rec = np.squeeze(rec)

rec = butter_lowpass_filter(data=rec,cutoff=10000, fs=sample_rate)



model_path = '4_layer_globalpool.tflite'


# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


steps_to_make=int((duration-rec_duration)/sec_slide)



for i in range(1, steps_to_make+1):
    if i==1:
        window = rec[:int(sample_rate * rec_duration)]

    else:
        window=rec[int(i*sec_slide*sample_rate):int((sample_rate*(rec_duration+i*sec_slide)))]

    window_mfcc=mfcc(y=window, sr=sample_rate, n_mfcc=20, n_fft=512)

    window_mfcc_z=ZScore(window_mfcc)

    window_mfcc_z = np.pad(window_mfcc_z, pad_width=((0, 0), (0, 1)), mode='constant')


    in_tensor = np.float32(window_mfcc_z.reshape(1, (window_mfcc_z.shape[0]), window_mfcc_z.shape[1], 1))

    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data= np.around(output_data[0], 2)


    location_dict={"Aufenthaltsraum":output_data[0], "Cafe":output_data[1],"Gruppenraum":output_data[2],
                   "Ruheraum":output_data[3], "Schlafzimmer":output_data[4], "Speisesaal":output_data[5]}





    print(str(i)+"th iteration of "+ str(steps_to_make)+ ":"+str(location_dict))

    if np.amax(output_data) > word_threshold:
        location_list=list(location_dict)


        print("####   " + location_list[np.argmax(output_data)] + "   ####")







