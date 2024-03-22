# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:02:55 2020

@author: onepiece
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt
from sklearn import metrics
import os
import soundfile as sf
import librosa
import torch
y1 = pd.read_csv('ff1010bird_metadata_2018.csv')

y2 =  pd.read_csv('warblrb10k_public_metadata_2018.csv')
fn1 = y1['itemid']
fn2 = y2['itemid']
fl1 = np.array(y1['hasbird'])
fl2 = np.array(y2['hasbird'] )
y_train = np.zeros([7690+8000,1]).ravel()
y_train[:7690] =fl1
y_train[7690:] = fl2 

array_of_audio = [] # this if for store all of the au data
# this function is for read image,the input is directory name
def read_directory(directory_name,fn):
    # this loop is for read each image in this foder,directory_name is the foder name with au.
    for filename in fn:
        fname = directory_name + str(filename) +".wav"
        # print(fname)
        
        audio_data,sr = sf.read(fname)
        h = 1/np.max(np.abs(audio_data))
        audio_data = audio_data * h
        kwargs_for_mel = {'n_mels': 40}
        x = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr, 
            n_fft=1024, 
            hop_length=512, 
            **kwargs_for_mel)
        m,n=np.shape(x)   
        if n<938:
            s = np.zeros((m,938-n))
            x= np.hstack((x,s))
        array_of_audio.append(np.transpose(x[:,: 938]))
        
       

    
direct1='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/2/ff1010/wav/'    
direct2='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/2/warbird/wav/'   
read_directory(direct1,fn1)    
# D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/2/warbird/wav/64486.wav
# sf.read('D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/2/ff1010/wav/55.wav')    
read_directory(direct2,fn2) 



direct3= 'D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/2/audio/'
t= np.load(direct3+str(0)+'.npy')

test = []
for i in range(0,4512):
    t= np.load(direct3+str(i)+'.npy')
    h = 1/np.max(np.abs(t))
    audio_data = t * h
    kwargs_for_mel = {'n_mels':40}
    x = librosa.feature.melspectrogram(
    y=audio_data, 
    sr=44100, 
    n_fft=1024, 
    hop_length=512, 
    **kwargs_for_mel)
    
    test.append(np.transpose(x[:,: 938]))



X_test=np.array(test)
X_train=np.array(array_of_audio)



#####################################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import  tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# T_train = np.reshape(X_train,[len(X_train),40*938])
# R_train = y_train
# t_train, t_test, r_train, r_test = train_test_split(T_train, R_train, test_size=0.05) 
# T_test = np.reshape(X_test,[len(X_test),40*938])
# clf = RandomForestClassifier(n_estimators=1000)
# clf.fit(t_train, r_train)
# yp = clf.predict(t_test)
# ac = metrics.accuracy_score(r_test,yp)
# ypp  = clf.predict_proba(T_test)
# # print(ac*100)
# print('Result for', t,'trees')
# print('------------------------------')
# print('Accuracy is ', str(ac*100),'%')
# # print('prob is ', str(ypp[]*100),'%')
# with open("sample_submission_randomforest.csv", "w") as fp: 
#     fp.write("ID,Predicted\n") 
#     for idx in range(4512): 
#         a = ypp[idx][1]
#         fp.write(f"{idx:05},{a}\n") 


##################
train, test, l_train,l_test = train_test_split(X_train, y_train, test_size=0.1)   
l_train = keras.utils.to_categorical(l_train, 2)
l_test = keras.utils.to_categorical(l_test, 2)
# SAve var on harddisk
import joblib
# 保存x
joblib.dump(X_train, 'X_train.pkl') 
joblib.dump(y_train, 'y_train.pkl') 
joblib.dump(X_test, 'X_test.pkl')
# 加载x
# x = joblib.load('x.pkl') 


# layers2 = [keras.Input(shape=(40,938)),
#     tf.keras.layers.LSTM(units = 128, return_sequences = True,activation='tanh'),
#     Dropout(rate=0.2),
#     tf.keras.layers.LSTM(units = 64, return_sequences = True,activation='tanh'),
#     Dropout(rate=0.2),
#     tf.keras.layers.LSTM(units = 128, return_sequences = True,activation='tanh'),
#     tf.keras.layers.Flatten(),
# tf.keras.layers.Dense(2, activation = 'softmax')]

# model2 = Sequential(layers2)
# model2.summary()
# model2.compile(optimizer=tf.keras.optimizers.Adam(
#     learning_rate=0.0001), loss='categorical_crossentropy', metrics=[tf.keras.metrics.AUC()])
# model2.fit(train, l_train,
#            batch_size=(32),
#           epochs=30,
#           # verbose=1,
#           validation_data=(test, l_test))

# ypt2 = model2.predict_proba(X_test)
# with open("sample_submission_rnn.csv", "w") as fp: 
#     fp.write("ID,Predicted\n") 
#     for idx in range(4512): 
#         a = ypt2[idx][1]
#         fp.write(f"{idx:05},{a}\n") 
# with open("sample_submission_rnn2.csv", "w") as fp: 
#     fp.write("ID,Predicted\n") 
#     for idx in range(4512): 
#         a = ypt2[idx][0]
#         fp.write(f"{idx:05},{a}\n")      


##################################################
# # model3 = []
# layers3 = [keras.Input(shape=(40,938)),
            
#     tf.keras.layers.LSTM(units = 938,return_sequences = True,time_major = True),
#   tf.keras.layers.LSTM(units = 512,return_sequences = True,time_major = True),
#   tf.keras.layers.LSTM(units = 256,return_sequences = True,time_major = True),
#   tf.keras.layers.Conv1D(kernel_size = 5 , filters = 32, activation='relu',padding='same'),
#   tf.keras.layers.Conv1D(kernel_size = 5 , filters = 16 , activation='relu',padding='same'),
  
 
#   tf.keras.layers.Flatten(),
  
# tf.keras.layers.Dense(2, activation = 'softmax')]

# model3 = Sequential(layers3)
# model3.summary()
# model3.compile(optimizer=tf.keras.optimizers.Adam(
#     learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='ROC')])
# model3.fit(train, l_train,
#             batch_size=(32),
#           epochs=10 ,
#           # verbose=1,
#           validation_data=(test, l_test))

# # ypt3 = model3.predict_proba(test)
# # acc = metrics.accuracy_score(l_test,ypt3)
# # print(acc)
# # ypt3 = model3.predict_proba(X_test)
# yt3 = model3.predict(X_test)
# ytc3=model3.predict_classes(X_test)
# with open("sample_submission_cnn_lstm.csv", "w") as fp: 
#     fp.write("ID,Predicted\n") 
#     for idx in range(4512): 
#         a = yt3[idx][1]
#         fp.write(f"{idx:05},{a}\n") 
# yyyyy = ypt3[:,0]+ypt3[:,1]


# Step 1: Check Pytorch (optional)
# £££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
# model4 = []
y_trainc = keras.utils.to_categorical(y_train, 2)
layers4 = [keras.Input(shape=(938,40)),
            
    # tf.keras.layers.LSTM(units = 1400,return_sequences = True,time_major = True),
  tf.keras.layers.LSTM(units = 64,return_sequences = True),
    tf.keras.layers.LSTM(units = 32,return_sequences = True),
    tf.keras.layers.LSTM(units = 32,return_sequences = True),
  tf.keras.layers.Conv1D(kernel_size = 5 , filters = 32, activation='relu'),
    tf.keras.layers.Conv1D(kernel_size = 3 , filters = 16 , activation='relu'),
  tf.keras.layers.Conv1D(kernel_size = 2 , filters = 8 , activation='relu'),
 
  tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(16 , activation = 'softmax'),
tf.keras.layers.Dense(2, activation = 'sigmoid')]

model4 = Sequential(layers4)
model4.summary()
model4.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='ROC')])
model4.fit(train, l_train,
          # batch_size=(16),
          epochs=5 ,
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

   