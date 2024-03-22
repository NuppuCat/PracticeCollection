# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:54:50 2020

@author: onepiece
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heapq
from collections import Counter


X = np.loadtxt('X_train.txt')
y = np.loadtxt('Y_train.txt')
tx = np.loadtxt('X_test.txt')
ty = np.loadtxt('Y_test.txt')
label =np.zeros([len(tx),1])
k = np.array([1,2,3,5,10,20])
for j in k:
        
    for i in range(len(tx)):
        distance=abs(X-tx[i])
        dis = (np.multiply(distance[:,0],distance[:,0])+np.multiply(distance[:,1],distance[:,1])+np.multiply(distance[:,2],distance[:,2]))
        min_list =  np.array(heapq.nsmallest(j, dis))
        # print(min_list)
        # ind = np.where(distance==np.min(distance))
        # print(ind)
        indx= np.zeros([len(min_list),1])
        for e in range(len(min_list)):
            
            indx1 = np.where(dis==min_list[e])
            indx[e] = indx1[0][0]
        # print(indx)
        # indx=ind[0][0]
        indx = np.int64(indx)
        a = y[indx]
        # c =np.array(Counter(a).most_common(1))
        # # print(c[0][0])
        # label[i]=c[0][0]
        c = np.zeros([len(a),1])
        for n in range(len(a)):
            c[n]=np.sum(a==a[n])
        l = np.where(c==np.max(c))[0][0]
        label[i]=a[l]
    # print(label)
    right_count=0
    for m in range(len(label)):
        
        right_count = right_count + (label[m]==ty[m])
    
    accuracy = right_count/len(tx)
    print('the accuracy with k=%d is %f' %(j,accuracy))
    label =np.zeros([len(tx),1])
