#import
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers import  MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from load_data import load_data
from keras.layers.advanced_activations import LeakyReLU  
from keras.models import model_from_json  
from keras.layers import *

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

import sys
import pdb


#create model
model = Sequential()

model.add(Convolution2D(4,5,5, border_mode='same',input_shape=(1,28,28))) #first conv layer
model.add(LeakyReLU(alpha=0.3))

model.add(Convolution2D(8,3,3, border_mode='same')) # second conv layer
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16,3,3, border_mode='same')) #third conv layer
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.3))
#model.add(LeakyReLU(alpha=0.3))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='Adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#load_data
data,label = load_data() # data=(42000,1,28,28)  label = (42000,10)

#label 0~9 10 keras need input_type is binary class matrices,so change
label = np_utils.to_categorical(label, 10)
#pdb.set_trace()

x_train, x_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size=0.3, random_state=1)

model.fit(x_train, y_train, nb_epoch=10)

score = model.evaluate(x_train, y_train, verbose=0) 
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=0) 
print('Test score:', score[0])
print('Test accuracy:', score[1])
