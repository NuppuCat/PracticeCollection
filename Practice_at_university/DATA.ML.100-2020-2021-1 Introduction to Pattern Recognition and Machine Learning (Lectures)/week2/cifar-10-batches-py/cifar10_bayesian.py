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
        # img_8x8 = resize(img, (8, 8))        
        img_1x1 = resize(img, (p, p)) 
        Img[i] = img_1x1
        r_vals = img_1x1[:,:,0].reshape(p*p)
        g_vals = img_1x1[:,:,1].reshape(p*p)
        b_vals = img_1x1[:,:,2].reshape(p*p)
        mu_r = r_vals.mean()
        mu_g = g_vals.mean()
        mu_b = b_vals.mean()
        X_mean[i,:] = (mu_r, mu_g, mu_b)
    return X_mean,Img
#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/data_batch_1')
datadict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
# print(Y)
Y=np.array(Y)
X = np.array(X)
def cifar10_color(X):
    
    p=1
    x_mean,img = convert_pic(X,p)
    
    
    
    return x_mean

def cifar10_2x2_color(X):
    
    p = 2
    x_mean,img = convert_pic(X,p)
    
    
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

def cifar10_classifier_naivebayes(x,mu,sigma,p):
    i=0
    c = np.zeros(np.size(x,axis=0))
    p = np.array(p)
    print(np.size(x,axis=0))
    while i<np.size(x,axis=0):
       
        py = stats.norm.pdf(x[i], mu, sigma)
        
        py = np.prod(py,axis=1)
        py = np.array(py)
        for j in range(0,np.size(py,axis=0)):
            py[j] = py[j]*p[j]
        py = py/np.sum(py)
        c[i]= np.where(py== np.max(py))[0][0]
        i = i + 1
    print(c)
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
    
def cifar10_classifier_bayes(x,mu,sigma,p):
    i=0
    c = np.zeros(np.size(x,axis=0))
    p = np.array(p)
    while i<np.size(x,axis=0):
        py = np.zeros(np.size(mu,axis=0))
       
        for h in range(0,np.size(mu,axis=0)):
            var = stats.multivariate_normal( mu[h], sigma[h])
            py[h] =  var.pdf(x[i])
            py[h] = py[h]*p[h]
        py = py/np.sum(py)
        c[i]= np.where(py== np.max(py))[0][0]
        i = i + 1
    print(c)
    return c    
    
    
    


def class_acc(pred,gt):
    accV = np.zeros(np.size(pred))
    for i in range(0, np.size(pred)) : 
        accV[i] = (pred[i]==gt[i])
    acc = sum(accV==1)/len(accV)*100
    return acc







xf = cifar10_color(X)
mu,sigma,p = cifar_10_naivebayes_learn(xf,Y)
c = cifar10_classifier_naivebayes(xf,mu,sigma,p)
print(class_acc(c,Y))


mu2,sigma2,p2 =cifar_10_bayes_learn(xf,Y)
c2 = cifar10_classifier_bayes(xf,mu2,sigma2,p2)
print(class_acc(c2,Y))




xf2 = cifar10_2x2_color(X)





# print(len(xf2))
#--------------------------
#     # Show some images randomly
#     if random() > 0.999:
#         plt.figure(1);
#         plt.clf()
#         plt.imshow(img_8x8)
#         plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
#         plt.pause(1)
