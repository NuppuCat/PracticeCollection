# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:52:32 2020

@author: onepiece
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,dct
import librosa
from scipy import signal
import librosa.display
import soundfile as sf
import sounddevice as sd
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import itertools

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
#  Step1

data = pd.read_csv('050455799.csv')
fileNames  = data["fileName"]
tags  = data["tags"]
direct1='D:/GRAM/MasterProgramme/Tampere/COMP.SGN.120 Audio intro/project/'  
MFCC =[]
#Ssave avg and std
features =[]
for f in fileNames:
    fname = direct1 + str(f[:(len(f)-3)]) +'wav'
    # print(fn)
    audio_data,sr = sf.read(fname)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, S=None, n_mfcc=40)
    MFCC.append(mfcc)
    m = np.mean(mfcc,axis=1)
    s = np.std(mfcc,axis=1)
    f =  np.linspace(0,80,80)
    f[:40]=m
    f[40:]=s
    features.append(f)
S = np.zeros((len(features),len(features)))   
D = np.zeros((len(features),len(features)))  
for i in range(len(features)):
    for j in range(len(features)):
        #改变维度为1行、d列 （-1表示列数自动计算，d= a*b /m ）
        S[i,j]=cosine_similarity(features[i].reshape(1,-1),features[j].reshape(1,-1))
        #DTW way
        distance, path = fastdtw(MFCC[i], MFCC[j], dist=euclidean)    
        D[i,j]=distance




import joblib
# save the var
joblib.dump(S, 'S.pkl') 
joblib.dump(D, 'D.pkl') 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Step 2
tag_label =[]
for tag in tags:
    if type(tag) == str:
        
        ls =tag.split(',')
        for l in ls:
          tag_label.append(l)  
# get all label
tag_label = pd.unique(tag_label).tolist() 
M = np.zeros(len(tag_label))   
MD= np.zeros(len(tag_label))
feature_orderbyclass =[]
for i in range(len(tag_label)):
    #save index in class
    f_in_class =[]
    
    for j in  range(len(features)):
        if type(tags[j])==str:
            if tag_label[i] in tags[j]:
                f_in_class.append(j)
    feature_orderbyclass.append(f_in_class)           
     #save all S(i,j) have label 
    s = []     
    d = []     
    if  f_in_class != []:
         if len(f_in_class) !=1:
             bb = list(itertools.permutations(f_in_class, 2))
             for b in bb:
                 s.append(S[b[0],b[1]])
                 d.append(D[b[0],b[1]])
             M[i]=np.mean(s)
             MD[i]=np.mean(d)
         # if only 1 sample in class, set its error mean as 1    
         else:   
             M[i]=1 
             MD[i]=1
    #if no sample in class, set its error mean as 0    
    else: 
        MD[i]=0
        M[i] = 0     
#  print two ways' average vector
print(M)
print(MD)

avg1 = np.mean(np.mean(S))
avg2 = np.mean(np.mean(D))

# plot 2 average 
plt.figure()
plt.plot(M)
plt.show()
plt.figure()
plt.plot(MD)
plt.show()
print(avg1)
print(avg2)

# draw heatmap
import seaborn as sns
plt.figure()
sns.heatmap(pd.DataFrame(S, columns = range(len(features)), index =  range(len(features))), 
                xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
plt.show()
plt.figure()
sns.heatmap(pd.DataFrame(D/np.max(D), columns = range(len(features)), index =  range(len(features))), 
                xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
plt.show()
#很明显的深色线表明与其它类的不同，对应来看输于不同场合
# 深色线相交的的地方颜色明显要浅，这说明它们可能来自同一场景，或者有同一类内容出现
# draw heatmap, to observe one class vs all set.
for i in range(len(feature_orderbyclass)):
    
    plt.figure()
    sns.heatmap(pd.DataFrame(D[:,feature_orderbyclass[i]], index = range(len(features)),
                             columns =  range(np.shape(D[:,feature_orderbyclass[i]])[1])), 
                    xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
    plt.title(tag_label[i], fontsize = 18)
    plt.show()
# draw heatmap, to observe one class.
for i in range(len(feature_orderbyclass)):
    smalld = D[:,feature_orderbyclass[i]]
    smalld = smalld[feature_orderbyclass[i],:]
    plt.figure()
    sns.heatmap(pd.DataFrame(smalld,
                             index = range(np.shape(D[:,feature_orderbyclass[i]])[1]),
                             columns =  range(np.shape(D[:,feature_orderbyclass[i]])[1])), 
                    xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
    plt.title(tag_label[i], fontsize = 18)
    plt.show()

# calculate the number of each class
print(tag_label)
for fo in feature_orderbyclass:
    print(len(fo))
# calculate the average number of labels per sample
tag_label2 =[]
for tag in tags:
    if type(tag) == str:
        
        ls =tag.split(',')
        for l in ls:
          tag_label2.append(l) 
print(len(tag_label2)/len(tags))




