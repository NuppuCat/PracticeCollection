# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:44:17 2020

@author: onepiece
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:05:38 2020

@author: onepiece
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
import scipy.stats as stats

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict
def convert_pic(X,p):
    X_mean = np.zeros((len(X),3))
    Img = np.zeros((len(X),p,p,3))
    for i in range(X.shape[0]):
        # Convert images to mean values of each color channel
        img = X[i]
        img_1x1 = resize(img, (p, p)) 
        Img[i] = img_1x1
    return Img

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
XW = XW.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
YW = np.array(YW)






datadict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y=np.array(Y)
X = np.array(X)
def cifar10_color(X):
    
    p=1
    img = convert_pic(X,p)
    
    
    
    return img

def cifar10_2x2_color(X,p):
    
    # p = 2
    img = convert_pic(X,p)
    
    
    return img

def cifar_10_naivebayes_learn(Xf,Y):
    mu = np.zeros([10,np.size(Xf,axis=1)])
    sigma = np.zeros([10,np.size(Xf,axis=1)])
    p = np.zeros([10,1])
    for i in range(0,len(label_names)):
        indx = np.where(Y==i)
        mu[i] = np.mean(Xf[indx],axis=0)
        sigma[i] = np.var(Xf[indx],axis=0)
        p[i] = len(Xf[indx])/len(Xf)
    return mu,sigma,p

def cifar_10_naivebayes_learn_plus(Xf,Y,p):
    mu = np.zeros([10,p,p,3])
    sigma = np.zeros([10,p,p,3])
    # mu = np.zeros([10,np.size(Xf[0]),1])
    # sigma = np.zeros([10,np.size(Xf[0]),1])
    p = np.zeros([10,1])
    for i in range(0,len(label_names)):
        indx = np.where(Y==i)
        mu[i] = np.mean(Xf[indx],axis=0)
        sigma[i] = np.var(Xf[indx],axis=0)
        # mu[i] = np.reshape(np.mean(Xf[indx],axis=0),[np.size(Xf[0])])
        # sigma[i] = np.reshape( np.var(Xf[indx],axis=0),[np.size(Xf[0])])
        p[i] = len(Xf[indx])/len(Xf)
    return mu,sigma,p





def cifar10_classifier_naivebayes(x,mu,sigma,p):
    i=0
    c = np.zeros(np.size(x,axis=0))
    p = np.array(p)
    # print(np.size(x,axis=0))
    while i<np.size(x,axis=0):
       
        py = stats.norm.pdf(x[i], mu, sigma)
        
        py = np.prod(py,axis=1)
        py = np.array(py)
        for j in range(0,np.size(py,axis=0)):
            py[j] = py[j]*p[j]
        py = py/np.sum(py)
        c[i]= np.where(py== np.max(py))[0][0]
        i = i + 1
    # print(c)
    return c




def cifar_10_bayes_learn(Xf,Y):
    mu = np.zeros([10,np.size(Xf,axis=1)])
    sigma = np.zeros([10,np.size(Xf,axis=1),np.size(Xf,axis=1)])
    p = np.zeros([10,1])
    for i in range(0,len(label_names)):
        indx = np.where(Y==i)
        mu[i] = np.mean(Xf[indx],axis=0)
        sigma[i] = 1/len(Xf[indx])*(np.dot((Xf[indx]-mu[i]).T,(Xf[indx]-mu[i])))
        p[i] = len(Xf[indx])/len(Xf)
    
    return mu,sigma,p


def cifar_10_bayes_learn_plus(Xf,Y,p):
    Xf = np.reshape(Xf,[len(Xf),np.size(Xf[0])])
    # print(len(Xf))
    mu = np.zeros([10,np.size(Xf[0])])
    sigma = np.zeros([10,np.size(Xf[0]),np.size(Xf[0])])
    p = np.zeros([10,1])
    for i in range(0,len(label_names)):
        indx = np.where(Y==i)
        # mu[i] = (np.mean(Xf[indx],axis=0),np.size(Xf[0]))
        mu[i] = np.mean(Xf[indx],axis=0)
        # chat = np.reshape((Xf[indx]-mu[i]),[len(Xf[indx]),np.size(Xf[0])]).T
        # cha  = np.reshape((Xf[indx]-mu[i]),[len(Xf[indx]),np.size(Xf[0])])
        chat = (Xf[indx]-mu[i]).T
        cha = (Xf[indx]-mu[i])
        d = np.dot(chat,cha)
        # print(np.shape(chat))
        sigma[i] = 1/len(Xf[indx])*d
        p[i] = len(Xf[indx])/len(Xf)
    
    return mu,sigma,p

    
def cifar10_classifier_bayes(x,mu,sigma,p):
    i=0
    c = np.zeros(np.size(x,axis=0))
    p = np.array(p)
    x = np.reshape(x,[len(x),np.size(x[0])])
    while i<np.size(x,axis=0):
        py = np.zeros(np.size(mu,axis=0))
        # var = stats.multivariate_normal( mu, sigma)
        # py = var.pdf(x[i])
        for h in range(0,np.size(mu,axis=0)):
            var = stats.multivariate_normal( mu[h], sigma[h])
            py[h] =  var.pdf(x[i])
            py[h] = py[h]*p[h]
            # print(py[h])
        # while p>=16,np.sum(py) approx equal 0, without this step doesnt influence theresult.    
        # py = py/np.sum(py)
        
        idx = np.where(py== np.max(py))
        # print(np.max(py))
        if np.size(idx[0])!=0:
            c[i]= idx[0][0]
        # print(c[i])
        # c[i]= np.where(py== np.max(py))[0][0]
        i = i + 1
    # print(c)
    return c    
    
    
    


def class_acc(pred,gt):
    accV = np.zeros(np.size(pred))
    for i in range(0, np.size(pred)) : 
        accV[i] = (pred[i]==gt[i])
    acc = sum(accV==1)/len(accV)*100
    return acc







# xf = cifar10_color(X)
# mu,sigma,p = cifar_10_naivebayes_learn(xf,Y)
# c = cifar10_classifier_naivebayes(xf,mu,sigma,p)
# print(class_acc(c,Y))


# mu2,sigma2,p2 =cifar_10_bayes_learn(xf,Y)
# c2 = cifar10_classifier_bayes(xf,mu2,sigma2,p2)
# print(class_acc(c2,Y))

# xl = [1,2,4,8,16,32]
# the trainset scal is not enough to get the conv at 32 
xl = [1,2,4,8,16]
# xl = [32]
xl = np.array(xl)
yln = np.zeros(len(xl))
ylb = np.zeros(len(xl))
# for i in range(0,len(xl)):
#     xf2 = cifar10_2x2_color(X,xl[i])
#     # xf2t = xf2[range(100)]
#     xf2t = xf2
#     mu3,sigma3,p3 = cifar_10_naivebayes_learn_plus(xf2,Y,xl[i])
#     c3 = cifar10_classifier_naivebayes(xf2t,mu3,sigma3,p3)
#     print(class_acc(c3,Y))
#     yln[i] = class_acc(c3,Y)
#     mu4,sigma4,p4 = cifar_10_bayes_learn_plus(xf2,Y,xl[i])
#     c4 = cifar10_classifier_bayes(xf2t,mu4,sigma4,p4)
#     print(class_acc(c4,Y))
#     ylb[i] = class_acc(c4,Y)
    
for i in range(0,len(xl)):
    xf2 = cifar10_2x2_color(XW,xl[i])
    # xf2t = xf2[range(100)]
    xf2t = cifar10_2x2_color(X,xl[i])
    mu3,sigma3,p3 = cifar_10_naivebayes_learn_plus(xf2,YW,xl[i])
    c3 = cifar10_classifier_naivebayes(xf2,mu3,sigma3,p3)
    print(class_acc(c3,YW))
    yln[i] = class_acc(c3,YW)
    mu4,sigma4,p4 = cifar_10_bayes_learn_plus(xf2,YW,xl[i])
    c4 = cifar10_classifier_bayes(xf2,mu4,sigma4,p4)
    print(class_acc(c4,YW))
    ylb[i] = class_acc(c4,YW)
plt.figure(1);
plt.plot(xl,yln,'r')
plt.plot(xl,ylb,'b')




# xf2 = cifar10_2x2_color(X,4)

# mu3,sigma3,p3 = cifar_10_naivebayes_learn_plus(xf2,Y,4)
# c3 = cifar10_classifier_naivebayes(xf2,mu3,sigma3,p3)
# print(class_acc(c3,Y))

# mu4,sigma4,p4 = cifar_10_bayes_learn_plus(xf2,Y,4)
# c4 = cifar10_classifier_bayes(xf2,mu4,sigma4,p4)
# print(class_acc(c4,Y))

# print(len(xf2))
#--------------------------
#     # Show some images randomly
#     if random() > 0.999:
#         plt.figure(1);
#         plt.clf()
#         plt.imshow(img_8x8)
#         plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
#         plt.pause(1)
