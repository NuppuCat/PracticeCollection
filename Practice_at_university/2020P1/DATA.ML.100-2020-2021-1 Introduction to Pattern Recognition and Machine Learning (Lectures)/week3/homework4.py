# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:58:52 2020

@author: onepiece
"""
# import pickle
import numpy as np
# import matplotlib.pyplot as plt
# from random import random
def minCluster(X):
    it = 1
    cluster = list()
   
    while True:
        if np.max(X)==np.min(X):
            break
        print("step",it)
        a,b = np.where(X==np.min(X))
        
        # print(len(a))
        if len(a)>1:
            print(a,b)
            for i in range(0,len(a)):
                # print(X)
                print(a[i]+1," and ",b[i]+1,"in one cluster")
                cluster.append([a[i]+1,b[i]+1])
        else:
             print(a+1," and ",b+1,"in one cluster")
             cluster.append([a[0]+1,b[0]+1])
        
        X[a,b] =  np.max(X)
        it=it+1 
    
    cluster= np.array(cluster)
    # print(cluster)
   
        
        
a1  = [0,4,13,24,12,8]
a2 = [0,0,10,22,11,10]
a3 = [0,0,0,7,3,9]
a4 = [0,0,0,0,6,18]
a5 = [0,0,0,0,0,8.5]
a6 = np.zeros(6)

X = np.zeros([6,6])
X[0] = a1
X[1] = a2
X[2] = a3
X[3] = a4
X[4] = a5
X[5] = a6
X[np.where(X==0)]=np.max(X)+1
# print(X)
minCluster(X)

