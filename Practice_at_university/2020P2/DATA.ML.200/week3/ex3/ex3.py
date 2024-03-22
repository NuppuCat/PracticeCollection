# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:18:47 2020

@author: onepiece
"""

import cv2
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage import color
import os
from sklearn.model_selection import train_test_split

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        # print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        # gray  = color.rgb2gray(img) 
        img2 = cv2.resize(img,(32,32))
        array_of_img.append(img2)
        #print(img)
       
# array_of_img = np.array(array_of_img)
for i in range(0,9):
    # print(i)
    direct='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/week3/ex3/0000'+str(i)
    # direct='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/1/t'
    read_directory(direct)
X = np.array(array_of_img)
y = np.ones([len(X),1]).ravel()


X = np.reshape(X,[len(X),32*32*3])
y[:150]=0
y[150:1500+150]=1
y[1500+150:1500+150+1500]=2
y[1500+150+1500:1500+150+1500+960]=3
y[4110:4110+1320]=4
y[5430:5430+1260]=5
y[6690:6690+300]=6
y[6990:6990+960]=7
y[7950:7950+960]=8
y = y.ravel()
# set this to get stable random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = []
from sklearn.neighbors import KNeighborsClassifier
model1 = KNeighborsClassifier(n_neighbors=3)
models.append(model1)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model2 = LinearDiscriminantAnalysis()
models.append(model2)
from sklearn.linear_model import LogisticRegression
model3 = LogisticRegression()
models.append(model3)
from sklearn import svm
model4 = svm.SVC(kernel='linear')
models.append(model4)
model5 = svm.SVC(kernel='rbf')
models.append(model5)


from sklearn.ensemble import RandomForestClassifier
model6 = RandomForestClassifier(n_estimators=20)
models.append(model6)

import time

from sklearn import metrics
from sklearn.model_selection import cross_val_score
t = ['3NN','LDA','LR','SVML','SVMR','RF']
for j in range(len(models)):
    model = models[j]
    start_time = time.time()
    
    model.fit(X_train,y_train)
    elapsed_time = time.time() - start_time
    elapsed_time = elapsed_time/1000
    start_time2 = time.time()
    yp = model.predict(X_test)
    elapsed_time2 = time.time() - start_time2
    elapsed_time2 = elapsed_time2/len(yp)
    ac = metrics.accuracy_score(y_test,yp)
    scores = cross_val_score(model, X, y, cv=5)
    m = np.mean(scores)
    std = np.std(scores)
    print('Result for', t[j])
    print('------------------------------')
    print('Accuracy is ', str(ac*100),'%')
    print('Training time/batch ', str(elapsed_time),'s')
    print('Test time/sample ', str(elapsed_time2),'ms')
    print('Mean of cvs is',str(m))
    print('STD of cvs is',str(std))


X2 = np.reshape(X,[len(X),32,32,3])

X_trainn, X_testn, y_trainn, y_testn = train_test_split(X2, y, test_size=0.2)
import tensorflow as tf
import keras
num_classes = 9



y_trainn = keras.utils.to_categorical(y_trainn, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
layers = [tf.keras.layers.Conv2D(kernel_size = 3, filters = 32, activation='relu', input_shape=(32, 32, 3)),
tf.keras.layers.MaxPool2D(),
tf.keras.layers.Conv2D(kernel_size = 3, filters = 32, activation='relu'),
tf.keras.layers.MaxPool2D(),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(num_classes, activation = 'softmax')]
modelnn = tf.keras.Sequential(layers)
modelnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelnn.summary()
start_time = time.time()
modelnn.fit(X_trainn, y_trainn, epochs = 10)
elapsed_time = time.time() - start_time
elapsed_time = elapsed_time/1000
start_time2 = time.time()
ynn = modelnn.predict(X_testn)
elapsed_time2 = time.time() - start_time2
l = np.zeros([len(ynn),1]).ravel()
for i in range(len(ynn)):
    l[i] = np.argmax(ynn[i])
ac = metrics.accuracy_score(y_testn,l)    

print('Result for CNN')
print('------------------------------')
print('Accuracy is ', str(ac*100),'%')
print('Training time/batch ', str(elapsed_time),'s')
print('Test time/sample ', str(elapsed_time2),'ms')






