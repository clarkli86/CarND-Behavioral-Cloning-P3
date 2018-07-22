#!/usr/bin/python

import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Input, merge
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.callbacks import *
from keras.optimizers import Adam
from keras import backend as K

TRAINING_DATA_DIRS = ['training_data/basic_lap/', \
                      'training_data/basic_lap_clockwise/',
                      'training_data/recovery_lap/',
                      'training_data/recovery_lap_clockwise/',
                      'training_data/smooth_curves/',
                      'training_data/smooth_curves_clockwise/']


images = []
measurements = []

for TRAINING_DATA_DIR in TRAINING_DATA_DIRS:
    lines = []
    with open(TRAINING_DATA_DIR + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
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

def display_cropped(model, image):
    """
    Output from cropping layer. Will be useful for writeup
    """
    cropping_layer = model.get_layer('cropping2d_1')
    cropping_function = K.function([cropping_layer.input], [cropping_layer.output])
    cropped_image = cropping_function([image.reshape(1, 160, 320, 3)])[0]
    plt.title('Original')
    plt.imshow(image)
    plt.show()
    plt.title('Cropped')
    plt.imshow(np.uint8(cropped_image.reshape(70, 320, 3)))
    plt.show()

def traffic_net():
    net_input = Input(shape=(160, 320, 3))
    pool1 = net_input
    pool1 = Lambda(lambda x: (x / 255.0) - 0.5)(pool1)
    pool1 = Cropping2D(cropping=((65,25), (0,0)), input_shape=(3,160,320))(pool1)
    pool1 = Convolution2D(nb_filter=12, nb_row=3, nb_col=3, activation='relu', border_mode='valid')(pool1)
    pool1 = Convolution2D(nb_filter=24, nb_row=3, nb_col=3, activation='relu', border_mode='valid')(pool1)
    pool1 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(pool1)

    pool2 = Convolution2D(nb_filter=36, nb_row=5, nb_col=5, activation='relu', border_mode='valid')(pool1)
    pool2 = Convolution2D(nb_filter=48, nb_row=5, nb_col=5, activation='relu', border_mode='valid')(pool2)
    pool2 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(pool2)

    pool1 = Flatten()(pool1)
    pool2 = Flatten()(pool2)
    pools = merge([pool1, pool2], mode='concat', concat_axis=1)

    fc = Dense(512, activation='relu')(pools)
    fc = Dropout(0.5)(fc)
    fc = Dense(256, activation='relu')(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(1)(fc)

    model = Model(input=net_input, output=fc)
    return model

def nvdia_net():
    net_input = Input(shape=(160, 320, 3))
    pool1 = net_input
    pool1 = Lambda(lambda x: (x / 255.0) - 0.5)(pool1)
    pool1 = Cropping2D(cropping=((65,25), (0,0)), input_shape=(3,160,320))(pool1)
    pool1 = Convolution2D(nb_filter=24, nb_row=5, nb_col=5, sub_sample=(2, 2), activation='relu', border_mode='valid')(pool1)
    pool1 = Convolution2D(nb_filter=36, nb_row=5, nb_col=5, sub_sample=(2, 2), activation='relu', border_mode='valid')(pool1)
    pool1 = Convolution2D(nb_filter=48, nb_row=5, nb_col=5, sub_sample=(2, 2), activation='relu', border_mode='valid')(pool1)
    pool1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, sub_sample=(1, 1), activation='relu', border_mode='valid')(pool1)
    pool1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, sub_sample=(1, 1), activation='relu', border_mode='valid')(pool1)

    pool1 = Flatten()(pool1)

    fc = Dense(100, activation='relu')(pool1)
    fc = Dense(50, activation='relu')(pool1)
    fc = Dense(10, activation='relu')(pool1)
    fc = Dense(1)(fc)

    model = Model(input=net_input, output=fc)
    return model

model = traffic_net()
model.summary()
# TODO Exponential delay learning rate
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)
save_checkpointer = ModelCheckpoint(filepath="model.h5", monitor='val_loss', verbose=1, save_best_only=True)
stop_checkpointer = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, callbacks=[save_checkpointer, stop_checkpointer])
#display_cropped(model, X_train[0])
exit()
