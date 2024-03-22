import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict



#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/data_batch_1')
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



#set X Y as TEST set
X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)
# plot picture in train set
for i in range(XW.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(XW[i])
        plt.title(f"Image {i} label={label_names[YW[i]]} (num {YW[i]})")
        plt.pause(1)


def class_acc(pred,gt):
    # pred = ['asd','pwd']
    # gt = ['asd','qwe']
    accV = np.zeros(len(pred))
    for i in range(0, len(pred)) : 
        accV[i] = (pred[i]==gt[i])
    acc = sum(accV==1)/len(accV)*100
    # print(acc)
    return acc


def cifar10_classifier_random(x):
    a = np.random.randint(0,len(label_names),len(x))
    # print(len(labeldict))
    return a
def cifar10_classifier_1nn(x,trdata,trlabels):
    cost = np.zeros(len(trdata))
    bestLabel = np.zeros(len(x),dtype=int)
    for j in range(0,len(x)):
        # cost = np.sum( np.power(trdata-x[j],2) ,axis =1)
        cost = np.sum( (trdata-x[j])**2,axis =1)
        bestLabel[j] = trlabels[np.where(cost==np.min(cost))[0][0]]
        # print(j)
    
    return bestLabel
def plot_picture_with_label(ind,lab,s):
    if s == 'X':
        for i in range(0,len(ind)):
            plt.figure(2);
            plt.clf()
            
            plt.imshow(X[ind[i]])
            plt.title(f"Image {ind[i]} predict_label={label_names[lab[i]]}(num {lab[i]}) label={label_names[Y[ind[i]]]} (num {Y[ind[i]]})")
            plt.pause(1)
    elif s == 'XW':
        for i in range(0,len(ind)):
            plt.figure(2);
            plt.clf()
            
            plt.imshow(XW[ind[i]])
            plt.title(f"Image {ind[i]} predict_label={label_names[lab[i]]}(num {lab[i]}) label={label_names[YW[ind[i]]]} (num {YW[ind[i]]})")
            plt.pause(1)
    else:return 0
        

#pick 10 random index samples in whole set. 
testIdx = np.random.randint(0,len(X),100)
# pick whole testset
# testIdx = range(0,len(X))

# test random classifier
y = cifar10_classifier_random(X[testIdx])
plot_picture_with_label(testIdx,y,'X')
print(f"{y} and {Y[testIdx]}")
accRandom = class_acc(y,Y[testIdx])
print(f"the accurate rate of random classifier in this time is {accRandom}%")


# test 1nn classifier
label = cifar10_classifier_1nn(X[testIdx],XW,YW)
plot_picture_with_label(testIdx,label,'X')
print(f"{label} and {Y[testIdx]}")
accNN1 = class_acc(label,Y[testIdx])
print(f"the accurate rate of NN1 classifier in this time is {accNN1}%")

# test 1nn classifier with x in train set
testIdxt = np.random.randint(0,len(XW),100)
labelt = cifar10_classifier_1nn(XW[testIdxt],XW,YW)
plot_picture_with_label(testIdxt,labelt,'XW')
print(f"{labelt} and {YW[testIdxt]}")
accNN1t = class_acc(labelt,YW[testIdxt])
print(f"the accurate rate of NN1 classifier which x in trian set is {accNN1t}%")
