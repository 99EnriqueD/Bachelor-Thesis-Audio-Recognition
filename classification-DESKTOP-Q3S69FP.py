from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.optimizers import Adam, SGD, Adamax
from keras.utils import np_utils
from sklearn import metrics
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import datetime


def reSample(data, samples):
    r = len(data)/samples  # re-sampling ratio
    newdata = []
    for i in range(0, samples):
        newdata.append(data[int(i*r)])
    return np.array(newdata)


train_subjects = ['s01', 's02', 's03']
validation_subjects = ['s04']
test_subjects = ['s05']
name_array = []

def get_data(path, sampleSize):

    #mergedActivities = ['Drinking', 'Eating', 'LyingDown', 'OpeningPillContainer',
    #                    'PickingObject', 'Reading', 'SitStill', 'Sitting', 'Sleeping',
    #                    'StandUp', 'UseLaptop', 'UsingPhone', 'WakeUp', 'Walking',
    #                    'WaterPouring', 'Writing']

    #specificActivities = ['Calling', 'Clapping',
    #                      'Falling', 'Sweeping', 'WashingHand', 'WatchingTV']
    specificActivities = ['GlassBreak', 'Scream', 'Crash', 'WaterSounds', 'Other']

    enteringExiting = ['Entering', 'Exiting']


    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_validation = []
    Y_validation = []

    # Note that 'stft_257_1' contains the STFT features with specification specified in the medium article;
    # https://medium.com/@chathuranga.15/sound-event-classification-using-machine-learning-8768092beafc

    for file in os.listdir(path + 'stft_257_1/'):

        a = (np.load(path + "stft_257_1/" + file)).T
        label = file.split('_')[-1].split(".")[0]
        if(label in specificActivities):
            # if(a.shape[0]>100 and a.shape[0]<500):
            if file.split("_")[0] in train_subjects:
                # X_train.append(reSample(a,sampleSize))
                X_train.append(np.mean(a, axis=0))
                Y_train.append(label)
            elif file.split("_")[0] in validation_subjects:
                X_validation.append(np.mean(a, axis=0))
                Y_validation.append(label)
            else:
                name_array.append(file)
                X_test.append(np.mean(a, axis=0))
                Y_test.append(label)
                # samples[label].append(reSample(a,sampleSize))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test


def print_M(conf_M):
    s = "activity,"
    for i in range(len(conf_M)):
        s += lb.inverse_transform([i])[0] + ","
    print(s[:-1])
    for i in range(len(conf_M)):
        s = ""
        for j in range(len(conf_M)):
            s += str(conf_M[i][j])
            s += ","
        print(lb.inverse_transform([i])[0], ",", s[:-1])
    print()


def print_M_P(conf_M):
    s = "activity,"
    for i in range(len(conf_M)):
        s += lb.inverse_transform([i])[0] + ","
    print(s[:-1])
    for i in range(len(conf_M)):
        s = ""
        for j in range(len(conf_M)):
            val = conf_M[i][j]/float(sum(conf_M[i]))
            s += str(round(val, 2))
            s += ","
        print(lb.inverse_transform([i])[0], ",", s[:-1])
    print()


def showResult():
    predictions = [np.argmax(y) for y in result]
    expected = [np.argmax(y) for y in y_test]
    conf_M = []
    num_labels = y_test[0].shape[0]
    for i in range(num_labels):
        r = []
        for j in range(num_labels):
            r.append(0)
        conf_M.append(r)

    n_tests = len(predictions)
    for i in range(n_tests):
        conf_M[expected[i]][predictions[i]] += 1

    print_M(conf_M)
    print_M_P(conf_M)


featuresPath = "STFT_features/"

a, b, c, d, e, f = get_data(featuresPath, 250)


X_train, Y_train, X_validation, Y_validation, X_test, Y_test = a, b, c, d, e, f

n_samples = len(Y_train)
print("No of training samples: " + str(n_samples))
order = np.array(range(n_samples))
np.random.shuffle(order)
X_train = X_train[order]
Y_train = Y_train[order]

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(Y_train))
y_test = np_utils.to_categorical(lb.fit_transform(Y_test))
y_validation = np_utils.to_categorical(lb.fit_transform(Y_validation))
num_labels = y_train.shape[1]

num_labels = y_train.shape[1]
filter_size = 2

# build model
model = Sequential()

model.add(Conv1D(512, input_shape=257))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Conv1D(512))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Conv1D(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer=Adamax(learning_rate=0.002))
# model.summary()

model.fit(X_train, y_train, batch_size=5, epochs=60,
          validation_data=(X_validation, y_validation))

result = model.predict(X_test)

cnt = 0
for i in range(len(Y_test)):
    if(np.amax(result[i]) < 0.5):

        pred = np.argmax(result[i])
    else:
        pred = np.argmax(result[i])
    if np.argmax(y_test[i]) == pred:
        cnt += 1

acc = str(round(cnt*100/float(len(Y_test)), 2))
print("Accuracy: " + acc + "%")
showResult()

# save model (optional)
path = "Models/audio_NN_New" + \
    datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_json = model.to_json()
with open(path+"_acc_"+acc+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(path+"_acc_"+acc+".h5")
