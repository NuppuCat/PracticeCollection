# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:23:31 2020

@author: onepiece
"""
import  scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
import sklearn

# In[ ]:TASK 4
a = np.zeros(500)
n = np.array( range(0,100))
s = np.cos(2 * np.pi * 0.1 * n)
a2 = np.zeros(300)
# 拼接数组
y = np.concatenate((a,s,a2),axis=0)
y_n = y + np.sqrt(0.5) * np.random.randn(y.size)
h = np.exp(-2 * np.pi * 1j * 0.1 * n)
y2 = np.abs(np.convolve(h, y_n, 'same'))
y1 = np.convolve(s, y_n, 'same')
#这样可以每次都生成新图
fig, ax = plt.subplots(4, 1)
ax[0].plot(y)
ax[1].plot(y_n)
ax[2].plot(y1)
ax[3].plot(y2)
# In[ ]:TASK 5
mat = loadmat("twoClassData.mat")
X = mat["X"] # Collect the two variables. >>> 
y = mat["y"].ravel() #ravel变矩阵为数组
#取前200个样本
train = X[:200,...]
#取200以后的样本
test = X[200:,...]
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train,y[:200,...])
y1 = model.predict(test)
# ac3nn = model.score(test, y[200:,...])
# print(ac3nn)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model2 = LinearDiscriminantAnalysis()
model2.fit(train,y[:200,...])
y2 = model2.predict(test)
# aclda = model2.score(test, y[200:,...])
# print(aclda)
from sklearn.linear_model import LogisticRegression
model3 = LogisticRegression().fit(train,y[:200,...])
y3 = model3.predict(test)
# aclrc = model3.score(test, y[200:,...])
# print(aclrc)
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=20).fit(train,y[:200,...])
y4 = model4.predict(test)
# acrfc = model4.score(test, y[200:,...])
# print(acrfc)
from sklearn import metrics
ac1 = metrics.accuracy_score(y[200:,...],y1)
ra1 = metrics.roc_auc_score(y[200:,...],y1)
ac2 = metrics.accuracy_score(y[200:,...],y2)
ra2 = metrics.roc_auc_score(y[200:,...],y2)
ac3 = metrics.accuracy_score(y[200:,...],y3)
ra3 = metrics.roc_auc_score(y[200:,...],y3)
ac4 = metrics.accuracy_score(y[200:,...],y4)
ra4 = metrics.roc_auc_score(y[200:,...],y4)
from pandas import *
idx = Index(['3-Nearest Neighbor','Linear Discriminant Analysis','Logistic Regression','Random Forest (n estimators=20)'])
data = np.array([ac1,ra1,ac2,ra2,ac3,ra3,ac4,ra4])
data = np.reshape(data,[4,2])
columns=['Accuracy Score','ROC-AUC Score']
fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
the_table=plt.table(cellText=data, rowLabels=idx, colLabels=columns, 
                     colWidths = [0.1]*data.shape[1], loc='center',cellLoc='right')

the_table.set_fontsize(20)
the_table.scale(2.5,2.8)
plt.show()


from prettytable import PrettyTable
x= PrettyTable(['Model','Accuracy Score','ROC-AUC Score'])
x.add_row(['3-Nearest Neighbor',ac1,ra1])
x.add_row(['Linear Discriminant Analysis',ac2,ra2])
x.add_row(['Logistic Regression',ac3,ra3])
x.add_row(['Random Forest (n estimators=20)',ac4,ra4])
print(x)









