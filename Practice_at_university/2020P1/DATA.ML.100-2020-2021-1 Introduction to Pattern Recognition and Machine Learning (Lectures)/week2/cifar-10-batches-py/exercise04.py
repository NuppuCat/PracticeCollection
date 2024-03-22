#!/usr/bin/env python
# coding: utf-8

# # Neural regression

# ### First we need to generate training data

# In[ ]:
import pickle
import numpy as np
import matplotlib.pyplot as plot
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
import scipy.stats as stats

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


datadict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/test_batch')




pathHead = 'D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/data_batch_'
datadicW = unpickle(f"{pathHead}1")

for i in range(2,6):
    # print(f"{pathHead}{i}")
    datadicti = unpickle(f"{pathHead}{i}")
    d = datadicW["data"]
    l = datadicW["labels"]
    datadicW["data"] = np.concatenate((d,datadicti["data"] ))
    datadicW["labels"] = np.concatenate((l,datadicti["labels"] ))
# SET XW YW as TRAIN SET
XW = datadicW["data"]
YW = datadicW["labels"]
XW = XW.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float64")
YW = np.array(YW)
yp = np.zeros([len(YW),10])
for i in range(0,10):
    idx = np.where(YW==i)
    # print(yp[idx[0],i])
    
    yp[idx[0],i]=1

# print(yp)
YW = yp
# print(np.shape(XW),np.shape(YW))

datadict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]
yp2 = np.zeros([len(Y),10])
for i in range(0,10):
    idx = np.where(Y==i)
    # print(yp[idx[0],i])
    
    yp2[idx[0],i]=1
Y = yp2
# print(yp2)
labeldict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float64")
Y=np.array(Y)
X = np.array(X)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
# Model sequential
model = Sequential()
# 1st hidden layer (we also need to tell the input dimension)
#   10 neurons, but you can change to play a bit
# model.add(Dense(5, input_shape(32,32,3), activation=’sigmoid’))
# model = keras.Sequential([
#     # keras.layers.Dense(5,input_shape=(32,32,3), activation='sigmoid'),
#     # keras.layers.Dense(128, activation='sigmoid'),
#     keras.layers.Conv2D(32,5,strides=2,input_shape=(32,32,3), activation="relu"),
#     keras.layers.Dense(10)
# ])
model.add(Dense(5, input_shape=(32,32,3), activation='sigmoid'))
# model.add(keras.Input(shape=(32, 32, 3)))
model.add(keras.layers.Conv2D(32,(5,5), strides=2, activation="relu"))
# model.add(keras.layers.Conv2D(64,(5,5), strides=2,input_shape=(32,32,3), activation="relu"))
model.add(keras.layers.Flatten())
# model.add(Dense(5, input_dim=3072, activation='sigmoid'))
## 2nd hidden layer - YOU MAY TEST THIS
#model.add(Dense(10, activation='sigmoid'))
# Output layer
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
# Learning rate has huge effect 
keras.optimizers.SGD(lr=0.01)
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])



model.fit(XW, YW, epochs=100)

test_loss, test_acc = model.evaluate(X,  Y, verbose=2)
print('\nTest accuracy:', test_acc)


# from sklearn.metrics import mean_squared_error 
# y_pred = model.predict(XW)
# print(YW[1])
# print(y_pred[1])
# print(np.sum(np.absolute(np.subtract(YW,y_pred)))/len(XW))
# print(np.square(np.subtract(YW,y_pred)).mean())
# print(len(YW))
# print(np.divide(np.sum(np.square(YW-y_pred)),len(YW)))
# print('MSE=',mean_squared_error(YW,y_pred))
# def class_acc(pred,gt):
#     accV = np.zeros(np.size(pred))
#     for i in range(0, len(pred)) : 
#         if ((pred[i]==gt[i]).all()):
#             accV[i] =1
#         else:
#             accV[i] =0    
        
#     acc = sum(accV==1)/len(accV)*100
#     return acc
# print( class_acc(y_pred,YW))


# ### Then we define NNet by adding suitable number of layers and neurons

# In[ ]:




# ### We train the network for number of epochs (10-10000, but you may test different values)

# In[ ]:




# ### Let's test how well the network models the data

# In[ ]:




# # Neural classification

# ### Let's make two classes in 2D

# In[ ]:


# Some random experiments with 2D Gaussians
# mu1 = [165,60]
# cov1 = [[10,0],[0,5]]
# mu2 = [180,80]
# cov2 = [[6,0],[0,10]]
# x1 = np.random.multivariate_normal(mu1, cov1, 100)
# x2 = np.random.multivariate_normal(mu2, cov2, 100)
# plot.plot(x1[:,0],x1[:,1],'rx')
# plot.plot(x2[:,0],x2[:,1],'gx')


# In[ ]:


# # Model sequential
# model2 = Sequential()
# # 1st hidden layer (we also need to tell the input dimension)
# model2.add(Dense(10, input_dim=2, activation='sigmoid'))
# ## 2nd hidden layer
#model.add(Dense(10, activation='sigmoid'))
# Output layer
#model.add(Dense(1, activation='sigmoid'))
# Output is 2D - [1 0] for class 1 and [0 1] for class 2
# model2.add(Dense(2, activation='sigmoid'))
# keras.optimizers.SGD(lr=0.1)
# model2.compile(optimizer='sgd', loss='mse', metrics=['mse'])


# # In[ ]:


# # Let's for the the 2D N input samples X and their 2D output labels Y
# X = np.row_stack((x1, x2))
# y1 = np.empty([x1.shape[0],2])
# y1[:,0] = 1
# y1[:,1] = 0
# y2 = np.empty([x2.shape[0],2])
# y2[:,0] = 0
# y2[:,1] = 1
# Y = np.row_stack((y1,y2))
# #print(Y)
# model2.fit(X, Y, epochs=100, verbose=1)


# # In[ ]:


# # You may check outputs for training data x1 (should be 1 0) and x2 (0 1)
# print(model2.predict(x1[0:9,:]))
# print(model2.predict(x2[0:9,:]))


# # In[ ]:


# # Let's plot how classification changes in different parts of the input space
# for xi in range(150,190,5):
#     for yi in range(50,100,5):
#         inp = np.empty([1,2])
#         inp[0,0] = xi
#         inp[0,1] = yi
#         cl_prob = model2.predict(inp)
#         if cl_prob[0][0] > cl_prob[0][1]:
#             plot.plot(xi,yi,'rx')
#         else:
#             plot.plot(xi,yi,'gx')

