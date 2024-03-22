# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:40:38 2020

@author: onepiece
"""
import numpy as np

pred = ['asd','pwd']
gt = ['asd','qwe']
accV= np.zeros(len(pred))
for i in range(0, len(pred)) : 
    accV[i] = (pred[i]==gt[i])
acc = sum(accV==1)/len(accV)

a = np.random.randint(0,999999,2)
c=np.array([20,30,40,50])
b = np.array([[1, 2, 3],
       [4, 5, 6]])
e= b[:,0]
d = np.array([[1, 2, 3]])
# d = np.power(b-c,2)
# f = sum(sum(b))
f =np.power( b-d,2)
f = np.sum(b,axis=1)
print(f)