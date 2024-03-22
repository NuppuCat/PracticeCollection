# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 08:40:15 2020

@author: onepiece
"""

import numpy as np
import matplotlib.pyplot as plt





X = np.loadtxt('X_train.txt')
y = np.loadtxt('Y_train.txt')
tx = np.loadtxt('X_test.txt')
ty = np.loadtxt('Y_test.txt')
label =np.zeros([len(tx),1])

for i in range(len(tx)):
    distance=abs(X-tx[i])
    dis = (np.multiply(distance[:,0],distance[:,0])+np.multiply(distance[:,1],distance[:,1])+np.multiply(distance[:,2],distance[:,2]))
    
    ind = np.where(dis==np.min(dis))
    # print(ind)
    indx=ind[0][0]
    label[i]=y[indx]
    # print(y[indx])
right_count=0
# print(label)
for j in range(len(label)):
    
    right_count = right_count + (label[j]==ty[j])

accuracy = right_count/len(tx)
print('the accuracy is',accuracy)
