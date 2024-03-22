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
    i = 1
    while True:
        if np.max(X)==np.min(X):
            break
        a,b = np.where(X==np.min(X))
        print("step",i)
        # print(len(a))
        if len(a)>1:
            print(a,b)
            for i in range(0,len(a)):
                # print(X)
                print(a[i]+1," and ",b[i]+1,"in one cluster")
                
           
        else:
             print(a+1," and ",b+1,"in one cluster")
        
        X[a,b] =  np.max(X)
        i=i+1 
        
a1  = [0,4,13,24,12,8]
a2 = [0,0,10,22,11,10]
a3 = [0,0,0,7,3,9]
a4 = [0,0,0,0,6,18]
a5 = [0,0,0,0,0,8.5]
a6 = np.zeros(6)
# a1.extend(a2)
# a1.extend(a3)
# a1.extend(a4)
# a1.extend(a5)
# a1.extend(a6)
# a1= np.reshape(a1,[6 ,6])
X = np.zeros([6,6])
X[0] = a1
X[1] = a2
X[2] = a3
X[3] = a4
X[4] = a5
X[5] = a6
X[np.where(X==0)]=np.max(X)+1
print(X)
minCluster(X)

