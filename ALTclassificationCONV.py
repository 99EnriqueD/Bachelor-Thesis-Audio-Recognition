
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Reshape
from keras.layers import Convolution2D, MaxPooling2D, MaxPooling1D, Conv1D, Conv2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, Adamax
from keras.utils import np_utils
from sklearn import metrics
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import datetime
from keras.layers.pooling import GlobalAveragePooling1D

"""
Script to train and test a convolutional neural network (CNN) model not using STFT-preprocessed data.

The model is saved so that it can be used in the SenCity sensor hub.
"""

train_subjects = ['s01', 's02', 's03']
validation_subjects = ['s04']
test_subjects = ['s05']
name_array = []
counters = {}
featuresPath = "NONSTFT_features/"

# Classes to train the model for
sound_classes = ['Glassbreak', 'Scream',
                 'Crash', 'Other',
                          'Watersounds']


def get_data(path):
    # Get training, validation, and testing data as well as the class weights.

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_validation = []
    Y_validation = []

    counters = {}
    total_count = 0
    for file in os.listdir(path + 'timeseries/'):

        a = (np.load(path + "timeseries/" + file)).T
        label = file.split('_')[-1].split(".")[0]
        if(label in sound_classes):
            if label in counters:
                counters[label] += 1
            else:
                counters[label] = 1
            total_count += 1
            if file.split("_")[0] in train_subjects:
                print(a.shape)
                print(a)
                X_train.append(a.tolist())
                Y_train.append(label)
            elif file.split("_")[0] in validation_subjects:
                X_validation.append(a)
                Y_validation.append(label)
            else:
                name_array.append(file)
                X_test.append(a)
                Y_test.append(label)

    X_train = np.array(X_train)
    print(X_train.shape)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)
    for label in counters:
        counters[label] = total_count/counters[label]
    weights = {}
    for i in range(len(sound_classes)):
        weights[i] = counters[sound_classes[i]]

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test, weights


def showResult():
    # Print confusion matrix

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


# Get training, validation, and testing data as well as the class weights.
a, b, c, d, e, f, weights = get_data(featuresPath)
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

KERNEL_SIZE = 10  # Filter height
INPUT_SHAPE = (None, 1)

# Build model
model = Sequential()

model.add(Conv1D(20, KERNEL_SIZE,
                 input_shape=INPUT_SHAPE, activation='relu'))
model.add(Conv1D(50, KERNEL_SIZE, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(120, KERNEL_SIZE, activation='relu'))
model.add(Conv1D(120, KERNEL_SIZE, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(num_labels))  # number of classes
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='Adam')

# Fit model
model.fit(np.expand_dims(X_train, axis=2), y_train, batch_size=5, epochs=480,
          validation_data=(np.expand_dims(X_validation, axis=2), y_validation), class_weight=weights, verbose=1)

# Print the model shapes and number of parameters.
print(model.summary())

result = model.predict(np.expand_dims(X_test, axis=2))

# Print model accuracy
cnt = 0
for i in range(len(Y_test)):
    # These prints can be used to see specific results of the test data.
    # print(result[i])
    # print(Y_test[i])
    if(np.amax(result[i]) < 0.5):
        pred = np.argmax(result[i])
    else:
        pred = np.argmax(result[i])
    if np.argmax(y_test[i]) == pred:
        cnt += 1
acc = str(round(cnt*100/float(len(Y_test)), 2))
print("Accuracy: " + acc + "%")

# Print confusion matrix
showResult()

# save model
path = "Models/audio_NN_ALTConv" + \
    datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_json = model.to_json()
with open(path+"_acc_"+acc+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(path+"_acc_"+acc+".h5")
