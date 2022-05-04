from Komprimiert import MFCC,MEL_SPEC, get_train_val_test, load_labels
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical

import wandb
import numpy as np
from Augmentation import Wav_Augmenter
from wandb.keras import WandbCallback
import csv

## NUR EINMAL VERWENDEN, FÜR TRAININGSDATEN:
#Wav_Augmenter.Wav_Data_Background(path="./TRAIN_DATA/", save_in="./TRAIN_DATA/")
#Wav_Augmenter.Wav_Data_Augment(path="./TRAIN_DATA/", save_in="./TRAIN_DATA/")

wandb.init(project="7-layer conv", entity="eckerdt")
config = wandb.config

config.max_len = MFCC.set_and_return_Max_length(65)
config.buckets = MFCC.set_and_return_Number_of_mfccs(20)

MFCC.save_data_and_scale(load_path="./TRAIN_DATA/", TRAIN_VAL_TEST="TRAIN_zScore_ohneWav", mfcc_zScore_lokal=True)
MFCC.save_data_and_scale(load_path="./VALIDATION_DATA/", TRAIN_VAL_TEST="VAL_zScore_ohneWav", mfcc_zScore_lokal=True)
MFCC.save_data_and_scale(load_path="./TEST_DATA/", TRAIN_VAL_TEST="TEST_zScore_ohneWav", mfcc_zScore_lokal=True)


X_train, Y_train = get_train_val_test(TRAIN_VAL_TEST="TRAIN_zScore_ohneWav",mfcc_or_mel_spec="mfcc")
X_val, Y_val = get_train_val_test(TRAIN_VAL_TEST="VAL_zScore_ohneWav",mfcc_or_mel_spec="mfcc")
X_test, Y_test = get_train_val_test(TRAIN_VAL_TEST="VAL_zScore_ohneWav",mfcc_or_mel_spec="mfcc")


#save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)
#X_train, X_test, y_train, y_test = get_train_test()




labels, _ = load_labels()

channels = 1
config.epochs = 300
num_classes = 6


X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len, channels)
X_val=X_val.reshape(X_val.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len, channels)


y_train_hot = to_categorical(Y_train)
y_val_hot = to_categorical(Y_val)
y_test_hot = to_categorical(Y_test)





model = Sequential()


#Layer 1
model.add(Conv2D(32,(3, 3),input_shape=(config.buckets, config.max_len, channels), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())


model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))

#Layer 2
#model.add(Conv2D(32, (3, 3), padding="same"))
#model.add(Activation("relu"))
#model.add(BatchNormalization())

#model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))

#Layer 3
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())


model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))

#Layer 4
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())


model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))

#Layer 5
#model.add(Conv2D(64, (3, 3), padding="same"))
#model.add(Activation("relu"))
#model.add(BatchNormalization())


#model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))

#Layer 6
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())


model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))

#Layer 7
#model.add(Conv2D(128, (3, 3), padding="same"))
#model.add(Activation("relu"))
#model.add(BatchNormalization())


#model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))



#model.add(Flatten())
model.add(GlobalMaxPooling2D())

model.add(Dense(875, activation='relu', kernel_regularizer="l2"))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])


#in 3 sets aufteilen training, validation und testing
hist=model.fit(X_train, y_train_hot, epochs=config.epochs, validation_data=(X_val, y_val_hot), callbacks=[WandbCallback()])


model.summary()

accuracy= np.around(np.mean(hist.history['acc']), 3)
val_accuracy= np.around(np.mean(hist.history['val_acc']), 3)
loss= np.around(np.mean(hist.history['loss']), 3)
val_loss= np.around(np.mean(hist.history['val_loss']), 3)



filepath="./Saved_CNN_models/"


model_name="4_Layer_875dense_globalMAXPool_mfcc_zScore_ohneWav_65len"


with open('./Model_accuracy.csv', 'a', newline='') as csv_write:
    writer = csv.writer(csv_write, delimiter=",")
    writer.writerow([model_name]+ [str(accuracy)]+[str(val_accuracy)]+[str(loss)]+[str(val_loss)])




model.save(filepath+model_name, overwrite=True)

