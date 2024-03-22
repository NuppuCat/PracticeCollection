# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:06:08 2020

@author: onepiece
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics

X_train=np.reshape(np.load('X_train.npy'),[4500,40*501])

X_test=np.reshape(np.load('X_test.npy'),[1500,40*501])
y_train=np.load('y_train.npy')
y_test=np.load('y_test.npy')
from sklearn.ensemble import RandomForestClassifier
tree=[10,50,100]

for t in tree:
    clf = RandomForestClassifier(n_estimators=t)
    clf.fit(X_train, y_train)
    yp = clf.predict(X_test)
    ypp = clf.predict_proba(X_test)
    ac = metrics.accuracy_score(y_test,yp)
    # print(ac*100)
    print('Result for', t,'trees')
    print('------------------------------')
    print('Accuracy is ', str(ac*100),'%')
    print('prob is ', str(ypp*100),'%')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import  tensorflow as tf
X_train=np.reshape(np.load('X_train.npy'),[4500,40*501])

X_test=np.reshape(np.load('X_test.npy'),[1500,40*501])
y_train=np.load('y_train.npy')
y_test=np.load('y_test.npy')
X_train = np.reshape(X_train,[4500,40,501])
X_train = X_train[..., np.newaxis]
X_test = np.reshape(X_test,[1500,40,501])
X_test = X_test[..., np.newaxis]
num_classes = 15
y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
layers = [tf.keras.layers.Conv2D(kernel_size = 5, filters = 32, activation='relu', input_shape=(40, 501, 1),padding='same'),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
tf.keras.layers.Conv2D(kernel_size = 5, filters = 32, activation='relu',padding='same'),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
tf.keras.layers.Flatten(),
Dense(10, activation = 'sigmoid'),
Dense(15, activation = 'sigmoid')]


model = Sequential(layers)
model.summary()
model.compile(optimizer=keras.optimizers.SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train,
          
          epochs=5)
          # verbose=1,
          # validation_data=(X_test, y_test))
ypt = model.predict(X_test)
l = np.zeros([len(ypt),1]).ravel()
for i in range(len(ypt)):
    l[i] = np.argmax(ypt[i])

ac = metrics.accuracy_score(y_test,l)

print('Accuracy is ', str(ac*100),'%')



X_train = np.reshape(X_train,[4500,40,501])
# X_train = X_train[..., np.newaxis]
X_test = np.reshape(X_test,[1500,40,501])
# X_test = X_test[..., np.newaxis]

y_train=np.load('y_train.npy')
y_test=np.load('y_test.npy')

layers2 = [keras.Input(shape=(40,501)),
    tf.keras.layers.LSTM(units = 32, return_sequences = True,activation='tanh'),
    tf.keras.layers.Flatten(),
tf.keras.layers.Dense(num_classes, activation = 'softmax')]
model2 = Sequential(layers2)
model2.summary()
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model2.compile(optimizer=keras.optimizers.SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_testt = keras.utils.to_categorical(y_test, num_classes)
model2.fit(X_train, y_train,
          
          epochs=30,
          # verbose=1,
          validation_data=(X_test, y_testt))
ypt2 = model2.predict(X_test)
l2 = np.zeros([len(ypt2),1]).ravel()
for i in range(len(ypt2)):
    l2[i] = np.argmax(ypt2[i])

ac2 = metrics.accuracy_score(y_test,l2)
print('Accuracy is ', str(ac2*100),'%')