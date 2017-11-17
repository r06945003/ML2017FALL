import tensorflow as tf
import keras.backend as k
from keras.backend.tensorflow_backend import set_session 

#save GPU resource-------------------------------------------------------------------
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

k.set_learning_phase(1)
from __future__ import print_function
import numpy as np  
import sys
#import pandas as pd  
import csv 
import math
import os
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LambdaCallback
from sklearn import cross_validation, ensemble, preprocessing, metrics
from keras.utils import np_utils


#loadpath = os.environ.get("GRAPE_DATASET_DIR")

x = []
y = []

n_row = 0
text = open(sys.argv[1], 'r') 
#text = open(loadpath + '/train.csv', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row != 0:
		y.append(r[0])
		r[1] = np.array(r[1].split(' '))
		r[1] = np.reshape(r[1], (1, 48, 48))
		x.append(r[1])
	n_row = n_row+1
text.close()
x = np.array(x)
y = np.array(y)
x = x.astype(np.float64)
x = x/255
y = y.astype(np.int)
y = np_utils.to_categorical(y, num_classes=7)


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    data_format='channels_first')

datagen.fit(x)

model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=3, input_shape=(1, 48, 48), padding='same', data_format='channels_first'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D((2,2), padding='same'))
#model.add(Dropout(0.2))

model.add(Convolution2D(filters=32, kernel_size=3, input_shape=(1, 48, 48), padding='same', data_format='channels_first'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D((2,2), padding='same'))
#model.add(Dropout(0.2))

model.add(Convolution2D(filters=32, kernel_size=3, input_shape=(1, 48, 48), padding='same', data_format='channels_first'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), padding='same'))
#model.add(Dropout(0.2))

model.add(Convolution2D(filters=64, kernel_size=3, input_shape=(1, 48, 48), padding='same', data_format='channels_first'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D((2,2), padding='same'))
#model.add(Dropout(0.2))

model.add(Convolution2D(filters=64, kernel_size=3, input_shape=(1, 48, 48), padding='same', data_format='channels_first'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D((2,2), padding='same'))
#model.add(Dropout(0.2))

model.add(Convolution2D(filters=64, kernel_size=3, input_shape=(1, 48, 48), padding='same', data_format='channels_first'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), padding='same'))
#model.add(Dropout(0.2))

model.add(Convolution2D(filters=128, kernel_size=3, input_shape=(1, 48, 48), padding='same', data_format='channels_first'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D((2,2), padding='same'))
#model.add(Dropout(0.2))

model.add(Convolution2D(filters=128, kernel_size=3, input_shape=(1, 48, 48), padding='same', data_format='channels_first'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D((2,2), padding='same'))
#model.add(Dropout(0.2))

model.add(Convolution2D(filters=128, kernel_size=3, input_shape=(1, 48, 48), padding='same', data_format='channels_first'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), padding='same'))
#model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

#filepath = "AUG_5.hdf5"

#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#batch_print_callback = LambdaCallback(on_epoch_end=lambda batch, logs:print('\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %(batch, logs['acc'], batch, logs['val_acc'])))

#callbacks_list = [checkpoint]
#model.fit(x, y, epochs=40, batch_size=128, callbacks=callbacks_list, validation_split=0.3)

#model.fit_generator(datagen.flow(x, y, batch_size=128),steps_per_epoch=len(x)//128, epochs=35)
model.fit_generator(datagen.flow(x, y, batch_size=128),samples_per_epoch=len(x), epochs=100)

model.save('AUG_final_8.hdf5')
#model.fit_generator(datagen.flow(x, y, batch_size=128),steps_per_epoch=len(x), epochs=35)


