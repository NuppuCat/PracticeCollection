# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:27:17 2020

@author: onepiece
"""
import cv2
import numpy as np
from matplotlib.image import imread
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
        gray  = color.rgb2gray(img) 
        array_of_img.append(gray)
        #print(img)
       
# array_of_img = np.array(array_of_img)
for i in range(1,11):
    # print(i)
    direct='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/1/train/train/'+str(i)
    # direct='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/1/t'
    read_directory(direct)
X = np.array(array_of_img)
y = np.ones([len(X),1])
for j in range(1,10):
    y[j*36000:]=j+1
y = y.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

X =  np.reshape(X,[len(X),28,28,1])
X_train = np.reshape(X_train,[len(X_train),28,28,1])
X_test = np.reshape(X_test,[len(X_test),28,28,1])


y_train[y_train==10]=0;
y_test[y_test==10]=0;


y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28,28,1)))
## 2nd hidden layer - YOU MAY TEST THIS
model.add(keras.layers.Conv2D(32,(3,3),  activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.2))
model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(10, activation='softmax'))
model.summary()
# model.add(Dense(5, input_dim=3072, activation='sigmoid'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.2),
              metrics=['accuracy'])
# Output layer
model.add(Dense(10, activation='softmax'))
# Learning rate has huge effect 
# keras.optimizers.SGD(lr=0.6)

# 训练模型
model.fit(X_train, y_train,
          # batch_size=3600,
          epochs=120,
          # verbose=1,
          validation_data=(X_test, y_test))



modelnn = Sequential()

modelnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28,28,1)))
## 2nd hidden layer - YOU MAY TEST THIS
modelnn.add(keras.layers.Conv2D(32,(3,3),  activation="relu"))
modelnn.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.2))
modelnn.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(10, activation='softmax'))
modelnn.summary()
# model.add(Dense(5, input_dim=3072, activation='sigmoid'))
modelnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.5),
              metrics=['accuracy'])
# Output layer
modelnn.add(Dense(10, activation='softmax'))
# Learning rate has huge effect 
# keras.optimizers.SGD(lr=0.6)

# 训练模型
modelnn.fit(X_train, y_train,
          # batch_size=3600,
          epochs=120,
          # verbose=1,
          validation_data=(X_test, y_test))


modelnn1 = Sequential()

modelnn1.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28,28,1)))
## 2nd hidden layer - YOU MAY TEST THIS
modelnn1.add(keras.layers.Conv2D(32,(3,3),  activation="relu"))
modelnn1.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.2))
modelnn1.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(10, activation='softmax'))
modelnn1.summary()
# model.add(Dense(5, input_dim=3072, activation='sigmoid'))
modelnn1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=1),
              metrics=['accuracy'])
# Output layer
modelnn1.add(Dense(10, activation='softmax'))
# Learning rate has huge effect 
# keras.optimizers.SGD(lr=0.6)

# 训练模型
modelnn1.fit(X_train, y_train,
          # batch_size=3600,
          epochs=120,
          # verbose=1,
          validation_data=(X_test, y_test))

modelnn2 = Sequential()

modelnn2.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28,28,1)))
## 2nd hidden layer - YOU MAY TEST THIS
modelnn2.add(keras.layers.Conv2D(64,(3,3),  activation="relu"))
modelnn2.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.2))
modelnn2.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(10, activation='softmax'))
modelnn2.summary()
# model.add(Dense(5, input_dim=3072, activation='sigmoid'))
modelnn2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=1),
              metrics=['accuracy'])
# Output layer
modelnn2.add(Dense(10, activation='softmax'))
# Learning rate has huge effect 
# keras.optimizers.SGD(lr=0.6)

# 训练模型
modelnn2.fit(X_train, y_train,
          # batch_size=3600,
          epochs=120,
          # verbose=1,
          validation_data=(X_test, y_test))















d2 = 'D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/1/test/test'
array_of_img2  = []
for filename in os.listdir(d2):
        # print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(d2 + "/" + filename)
        gray  = color.rgb2gray(img) 
        array_of_img2.append(gray)
Xt = np.array(array_of_img2)
X_t = np.reshape(Xt,[len(Xt),28,28,1])
yt = modelnn2.predict(X_t)
# yt = clf2.predict(X_t)
l = np.zeros([len(yt),1]).ravel()
for i in range(len(yt)):
    l[i] = np.argmax(yt[i])
l[l==0]=10;



with open("sample_submission.csv", "w") as fp: 
    fp.write("Id,Category\n") 
    for idx in range(10000): 
        a = int(l[idx])
        fp.write(f"{idx:05},{a}\n") 






















from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(n_jobs=-1 tol=1e-6 )
model = LogisticRegression(max_iter=10000,C=0.01,multi_class='ovr')
X =  np.reshape(X,[len(X),28*28])
X_train = np.reshape(X_train,[len(X_train),28*28])
X_test = np.reshape(X_test,[len(X_test),28*28])

model.fit(X,y)
y1 = model.predict(X_test)

from sklearn import metrics
ac1 = metrics.accuracy_score(y_test,y1)
print(ac1)

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(), LinearSVC())
clf.fit(X_train,y_train)
y2 = clf.predict(X_test)
ac2 = metrics.accuracy_score(y_test,y2)
print(ac2)
from sklearn.neural_network import MLPClassifier
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,) ,max_iter=100000).fit(X_train, y_train)
# clf2.fit(X_train,y_train)
y3 = clf2.predict(X_test)
ac3 = metrics.accuracy_score(y_test,y3)
print(ac3)










modeln1 = Sequential()

modeln1.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28,28,1)))
modeln1.add(Flatten())
modeln1.summary()
modeln1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.3),
              metrics=['accuracy'])
# keras.optimizers.SGD(lr=6)
modeln1.add(Dense(10, activation='softmax'))
modeln1.fit(X_train, y_train,
          # batch_size=3,
          epochs=1200,
          # verbose=1,
          validation_data=(X_test, y_test))


modeln = Sequential()

modeln.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28,28,1)))
modeln.add(Flatten())
modeln.summary()
modeln.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.95),
              metrics=['accuracy'])
# keras.optimizers.SGD(lr=6)
modeln.add(Dense(10, activation='softmax'))
modeln.fit(X_train, y_train,
          # batch_size=3,
          epochs=1200,
          # verbose=1,
          validation_data=(X_test, y_test))


