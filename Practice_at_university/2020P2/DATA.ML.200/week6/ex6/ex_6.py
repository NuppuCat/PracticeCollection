# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:42:04 2020

@author: onepiece
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt
from sklearn import metrics
import os
import soundfile as sf
import librosa
import torch
from scipy import io 
# In[3]:
data = io.loadmat('arcene.mat')

X_test = data['X_test']
X_train = data['X_train']
y_train = data['y_train'].ravel()
y_test = data['y_test'].ravel()

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
importances = clf.feature_importances_

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances,
        color="r")

plt.show()

yp = clf.predict(X_test)

ac = metrics.accuracy_score(y_test,yp)
print('accuracy is ',ac)
# In[4]:
from sklearn import feature_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
rfe = feature_selection.RFECV(estimator=LinearDiscriminantAnalysis(),step=50,verbose=1)
rfe.fit(X_train, y_train)
s=  np.count_nonzero(rfe.support_)
print(s,' selected features')
plt.figure()
plt.plot(range(0,10001,50), rfe.grid_scores_)
plt.show()
yp2 = rfe.predict(X_test)
ac2 = metrics.accuracy_score(y_test,yp2)
print('accuracy is ',ac2)
# In[5]:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
model3 = LogisticRegression(solver ='liblinear', penalty = 'l1')
C = [0.0001,0.001,0.01,0.1,1,10,100]
scores=[]
for c in C:
    model3.C = c
    # model3.fit(X_train, y_train)
    # prediction = clf.predict(X_test)
    # # accuracy = 100.0 * np.mean(prediction == y_test)
    score = cross_val_score(model3, X_train, y_train, cv = 5).mean()
    scores.append(score)
scores = np.array(scores)
bestc=C[np.argmax(scores)]
model3.C = bestc
model3.fit(X_train, y_train)
num =np.count_nonzero(model3.coef_) 
prediction = model3.predict(X_test)
accuracy = 100.0 * np.mean(prediction == y_test)
print('best C is',bestc,' and its accuracy is ',accuracy)

print('number of selected feature is',num)
