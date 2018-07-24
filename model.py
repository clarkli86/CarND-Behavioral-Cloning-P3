#!/usr/bin/python

import csv
import cv2
import sklearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Input, merge, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.callbacks import *
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split

TRAINING_DATA_DIRS = ['training_data/basic_lap/', \
                      'training_data/basic_lap_clockwise/',
                      'training_data/recovery_lap/',
                      #'training_data/recovery_lap_clockwise/',
                      'training_data/smooth_curves/',
                      'training_data/smooth_curves_clockwise/',
                      'training_data/basic_lap_2/',
                      'training_data/basic_lap_2_clockwise/',
                      'training_data/recovery_lap_2/']
                      #'training_data/recovery_lap_2_clockwise/']

def process_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def generator(samples, batch_size=32):
    """ Generate next batch of training data """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = process_image(batch_sample.filename)
                images.append(image)
                angles.append(batch_sample.steering)
                #print(batch_sample.filename + " angle: " + str(batch_sample.steering))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

images = []
measurements = []

class Sample(object):
    """Contains a sample and steering angle"""
    def __init__(self, filename, steering):
        self.filename = filename
        self.steering = steering

samples = []

for TRAINING_DATA_DIR in TRAINING_DATA_DIRS:
    with open(TRAINING_DATA_DIR + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            current_path = TRAINING_DATA_DIR + 'IMG/' + line[0].split('/')[-1]
            samples.append(Sample(current_path, float(line[3])))
            # Optionally use the other two cameras
            #current_path = TRAINING_DATA_DIR + 'IMG/' + line[1].split('/')[-1]
            #samples.append(Sample(current_path, min(1.0, float(line[3]) + 0.2)))
            #current_path = TRAINING_DATA_DIR + 'IMG/' + line[2].split('/')[-1]
            #samples.append(Sample(current_path, max(-1.0, float(line[3]) - 0.2)))

samples_left   = list(filter(lambda x : x.steering <= -0.4, samples))
samples_center = list(filter(lambda x : x.steering > -0.4 and x.steering < 0.4, samples))
samples_right  = list(filter(lambda x : x.steering >= 0.4, samples))

shuffle(samples_left)
shuffle(samples_center)
shuffle(samples_right)

train_samples = np.concatenate((samples_left[:int(len(samples_left) * 0.8)], samples_center[:int(len(samples_center) * 0.8)], samples_right[:int(len(samples_right) * 0.8)]))
validation_samples = np.concatenate((samples_left[int(len(samples_left) * 0.8):], samples_center[int(len(samples_center) * 0.8):], samples_right[int(len(samples_right) * 0.8):]))

# Plot distrubtion histograms in training/validation
fig, axes = plt.subplots(1, 2)
axes[0].hist([x.steering for x in train_samples], 10)
axes[0].set_title('Distribution in training')
axes[1].hist([x.steering for x in validation_samples], 10)
axes[1].set_title('Validation in training')
plt.show()

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

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
    pool1 = Cropping2D(cropping=((65,25), (0,0)), input_shape=(160, 320, 3))(pool1)
    pool1 = Convolution2D(nb_filter=12, nb_row=3, nb_col=3, border_mode='valid')(pool1)
    pool1 = ELU()(pool1)
    pool1 = Convolution2D(nb_filter=24, nb_row=3, nb_col=3, border_mode='valid')(pool1)
    pool1 = ELU()(pool1)
    pool1 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(pool1)

    pool2 = Convolution2D(nb_filter=36, nb_row=5, nb_col=5, border_mode='valid')(pool1)
    pool2 = ELU()(pool2)
    pool2 = Convolution2D(nb_filter=48, nb_row=5, nb_col=5, border_mode='valid')(pool2)
    pool2 = ELU()(pool2)
    pool2 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(pool2)

    pool1 = Flatten()(pool1)
    pool2 = Flatten()(pool2)
    pools = merge([pool1, pool2], mode='concat', concat_axis=1)

    fc = Dense(100)(pools)
    fc = ELU()(fc)
    fc = Dropout(0.2)(fc)
    fc = Dense(50)(fc)
    fc = ELU()(fc)
    fc = Dropout(0.2)(fc)
    fc = Dense(10)(fc)
    fc = ELU()(fc)
    fc = Dropout(0.2)(fc)
    fc = Dense(1)(fc)

    model = Model(input=net_input, output=fc)
    return model

def nvidia_net():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320, 3), output_shape=(160,320, 3)))
    model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=(160,320, 3)))
    model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))

    return model

model = nvidia_net()
#model = traffic_net()
model.summary()
# TODO Exponential delay learning rate
model.compile(loss='mse', optimizer='adam')
save_checkpointer = ModelCheckpoint(filepath="model.h5", monitor='val_loss', verbose=1, save_best_only=True)
stop_checkpointer = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=1, mode='auto')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10, callbacks=[save_checkpointer, stop_checkpointer])
#display_cropped(model, X_train[0])
exit()
