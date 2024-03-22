# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:25:02 2020

@author: onepiece
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt
from sklearn import metrics

def log_loss(w, X, y):
    """ 
    Computes the log-loss function at w. The 
    computation uses the data in X with
    corresponding labels in y. 
    """

    L = 0  # Accumulate loss terms here.
    
    # TODO: Sum up the loss for each sample in X to L
    for i in range(len(y)):
        L = np.log(1+np.exp(-y[i]*np.transpose(w)*X[i]))+L
    L = L+0.1*np.transpose(w)*w
    return L


def grad(w, X, y):
    """ 
    Computes the gradient of the log-loss function
    at w. The computation uses the data in X with
    corresponding labels in y. 
    """
    
    G = 0  # Accumulate gradient here.

    # TODO: Sum up the gradient for each sample in X to G
    for i in range(len(y)):
        G = -y[i]*X[i]+y[i]*X[i]/(1+np.exp(-y[i]*np.transpose(w)*X[i]))+G
    G = G+2*0.1*w
    return G


if __name__ == "__main__":

    # TODO: Add your code here:

    # 1) Load X and y data:
    
    X = np.loadtxt('X.csv', delimiter=',')
    y = genfromtxt('Y.csv')
    # 2) Initialize w at random:
    w = np.random.rand(2)
   
    # 3) Set step_size to a small positive value
    a  = 0.01
    # 4) Initialize empty lists for storing the path and
    # accuracies:
    w_array = [w]
    acl = [0]    
    for iteration in range(100):
        pass

        # 5) Apply the gradient descent rule:
        g = grad(w, X, y)    
        w = w-a*g;
        # 6) Print the current state:
        print(w)    
        # 7) Compute the accuracy:
        y1 =X@np.transpose(w)    
        y1[y1>0]=1
        y1[y1<0]=-1
        y1[y==0]=1
        
        ac = metrics.accuracy_score(y,y1)
        w_array.append(w)
        acl.append(ac)
    # 8) Below is a template for plotting. Feel free to
    # rewrite if you prefer different style:
    
        
    
    w_array = np.array(w_array)
    plt.figure(figsize=[5, 5])
    plt.subplot(211)
    plt.plot(w_array[:, 0], w_array[:, 1], 'ro-')
    plt.xlabel('w$_0$')
    plt.ylabel('w$_1$')
    plt.title('Optimization path')

    plt.subplot(212)
    plt.plot(100.0 * np.array(acl), linewidth=2)
    plt.ylabel('Accuracy / %')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig("log_loss_minimization.pdf", bbox_inches="tight")
# In[4]:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import  tensorflow as tf

layers = [tf.keras.layers.Conv2D(kernel_size = 5, filters = 32, activation='relu', input_shape=(64, 64, 3),padding='same'),
tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
tf.keras.layers.Conv2D(kernel_size = 5, filters = 32, activation='relu',padding='same'),
tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
tf.keras.layers.Flatten(),
Dense(100, activation = 'sigmoid'),
Dense(2, activation = 'sigmoid')]

model = Sequential(layers)
model.summary()
# In[5]:
model.compile(optimizer=keras.optimizers.SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

import cv2
import os
from sklearn.model_selection import train_test_split
from matplotlib.image import imread
array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        # print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        # gray  = color.rgb2gray(img) 
        img2 = cv2.resize(img,(64,64))
        array_of_img.append(img2)
        #print(img)
       
# array_of_img = np.array(array_of_img)
# here only use tow type to train and test,bcs the output layer is 2(as I think,or it will error)
for i in range(0,2):
    # print(i)
    direct='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/week3/ex3/0000'+str(i)
    # direct='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/1/t'
    read_directory(direct)
X = np.array(array_of_img)
y = np.ones([len(X),1]).ravel()


y[:150]=0
y[150:1500+150]=1
# y[1500+150:1500+150+1500]=2
# y[1500+150+1500:1500+150+1500+960]=3
# y[4110:4110+1320]=4
# y[5430:5430+1260]=5
# y[6690:6690+300]=6
# y[6990:6990+960]=7
# y[7950:7950+960]=8
y = y.ravel()
num_classes = 2
y = keras.utils.to_categorical(y, num_classes)
# set this to get stable random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    

model.fit(X_train, y_train,
          batch_size=32,
          epochs=20,
          # verbose=1,
          validation_data=(X_test, y_test))
