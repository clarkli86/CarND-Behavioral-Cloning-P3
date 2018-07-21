#!/usr/bin/python

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.engine.topology import Merge
from keras.layers.core import Dropout
from keras.callbacks import *
from keras.optimizers import Adam

TRAINING_DATA_DIR = 'training_data/basic_lap/'

lines = []
with open(TRAINING_DATA_DIR + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []
for line in lines:
    # Center image
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = TRAINING_DATA_DIR  + 'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    # Steering angle
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

def Traffic_Net():
    pool1 = Sequential()
    pool1.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    pool1.add(Convolution2D(nb_filter=12, nb_row=3, nb_col=3, activation='relu', border_mode='valid'))
    pool1.add(Convolution2D(nb_filter=24, nb_row=3, nb_col=3, activation='relu', border_mode='valid'))
    pool1.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    pool1.summary()
    pool1_flat = Sequential()
    pool1_flat.add(pool1)
    pool1_flat.add(Flatten())

    pool2 = Sequential()
    pool2.add(pool1)
    pool2.add(Convolution2D(nb_filter=36, nb_row=3, nb_col=3, activation='relu', border_mode='valid'))
    pool2.add(Convolution2D(nb_filter=48, nb_row=3, nb_col=3, activation='relu', border_mode='valid'))
    pool2.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    pool2.summary()
    pool2_flat = Sequential()
    pool2_flat.add(pool2)
    pool2_flat.add(Flatten())

    model = Sequential()
    model.add(Merge([pool1_flat, pool2_flat], mode='concat', concat_axis=1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

model = Traffic_Net()
model.summary()
# TODO Exponential delay learning rate
optimizer = Adam(lr=0.0005)
model.compile(loss='mse', optimizer=optimizer)
save_checkpointer = ModelCheckpoint(filepath="model.h5", monitor='val_loss', verbose=1, save_best_only=True)
stop_checkpointer = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1, callbacks=[save_checkpointer, stop_checkpointer])
exit()
