# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 08:40:15 2020

@author: onepiece
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




X = np.loadtxt('X_train.txt')
y = np.loadtxt('Y_train.txt')
tx = np.loadtxt('X_test.txt')
ty = np.loadtxt('Y_test.txt')
label =np.zeros([len(tx),1])

for i in range(len(tx)):
    distance=abs(X-i)
    # dis = (np.multiply(distance[:,0],distance[:,0])+np.multiply(distance[:,0],distance[:,0])+np.multiply(distance[:,0],distance[:,0]))
    
    ind = np.where(distance==np.min(distance))
    # print(ind)
    indx=ind[0][0]
    label[i]=y[indx]

right_count=0
for j in range(len(label)):
    
    right_count = right_count + (label[j]==ty[j])

accuracy = right_count/len(tx)
print('the accuracy is',accuracy)
