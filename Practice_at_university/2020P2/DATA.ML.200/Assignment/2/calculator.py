# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:12:19 2020

@author: onepiece
"""

import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from numpy import genfromtxt
# from sklearn import metrics
# import os
# import soundfile as sf
# import librosa
# import torch
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import keras
# from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import  tensorflow as tf
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

import joblib

import torch



# # 保存x
## joblib.dump(X_train, 'X_train.pkl') 
### joblib.dump(y_train, 'y_train.pkl') 
## joblib.dump(X_test, 'X_test.pkl')
## 加载x
X_train = joblib.load('X_train.pkl') 
y_train = joblib.load('y_train.pkl') 
X_test = joblib.load('X_test.pkl') 

train, test, l_train,l_test = train_test_split(X_train, y_train, test_size=0.1)   
l_train = keras.utils.to_categorical(l_train, 2)
l_test = keras.utils.to_categorical(l_test, 2)




y_trainc = keras.utils.to_categorical(y_train, 2)
layers4 = [keras.Input(shape=(938,40)),
         tf.keras.layers.Conv1D(kernel_size = 5 , filters = 16, activation='relu'),   
    # tf.keras.layers.LSTM(units = 1400,return_sequences = True,time_major = True),
   tf.keras.layers.LSTM(units = 8,return_sequences = True),
    tf.keras.layers.LSTM(units = 4,return_sequences = True),
    tf.keras.layers.LSTM(units = 2 ,return_sequences = True,time_major = True),
    tf.keras.layers.Conv1D(kernel_size = 2 , filters = 8 , activation='relu'),
  tf.keras.layers.Conv1D(kernel_size = 2 , filters = 4, activation='relu'),
 
  tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(16 , activation = 'softmax'),
tf.keras.layers.Dense(2, activation = 'sigmoid')]

model4 = Sequential(layers4)
model4.summary()
model4.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='ROC')])
model4.fit(train, l_train,
          # batch_size=(16),
          epochs= 10 ,
            # verbose=1,
            validation_data=(test, l_test))

# ypt3 = model3.predict_proba(test)
# acc = metrics.accuracy_score(l_test,ypt3)
# print(acc)
# ypt3 = model3.predict_proba(X_test)
yt4 = model4.predict(X_test)
print(np.max(np.max(yt4)))
# ytc4=model4.predict_classes(X_test)
with open("sample_submission_cnn_lstm_upgrade1.csv", "w") as fp: 
    fp.write("ID,Predicted\n") 
    for idx in range(4512): 
        a = yt4[idx][1]
        fp.write(f"{idx:05},{a}\n") 