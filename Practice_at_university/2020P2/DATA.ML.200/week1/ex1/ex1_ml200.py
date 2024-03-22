# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:11:19 2020

@author: onepiece
"""
import  scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
# In[ ]:TASK 3
mat = loadmat("twoClassData.mat")

print(mat.keys()) # Which variables mat contains? [’y’, ’X’, ’__version__’, ’__header__’, ’__globals__’] >>> 
X = mat["X"] # Collect the two variables. >>> 
y = mat["y"].ravel() 
#TAT first time know such skill
X0 = X[y == 0, :]
X1 = X[y == 1, :]
plt.figure(1)
plt.plot(X0[:, 0], X0[:, 1], 'ro')
plt.plot(X1[:, 0], X1[:, 1], 'bo')
# In[ ]:TASK 4
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

# Read the data

img = imread("uneven_illumination.jpg")
plt.figure(2)
plt.imshow(img, cmap='gray')
plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))
plt.show()

# Create the X-Y coordinate pairs in a matrix
# this create the matrix that [X,Y] is 坐标 
X, Y = np.meshgrid(range(1300), range(1030))
Z = img

x = X.ravel()
y = Y.ravel()
z = Z.ravel()

# ********* TODO 1 **********
# Create data matrix
# Use function "np.column_stack".
# Function "np.ones_like" creates a vector like the input.
# NOTICE: should set vectors in () 合并向量成矩阵
# 这个方法实际上是以过滤后的图片作为干扰，最大限度拟合“滤镜”的。但是如果图片本身和滤镜结合较深，那么个处理就会造成误差
H = np.column_stack((x**2,y**2,x*y,x,y,np.ones_like(x)))
# ********* TODO 2 **********
# Solve coefficients
# Use np.linalg.lstsq
# Put coefficients to variable "theta" which we use below.
theta = np.linalg.lstsq(H, z, rcond=None)[0]
# Predict
# @ is simple简化表达
z_pred = H @ theta
Z_pred = np.reshape(z_pred, X.shape)
plt.figure(3)
plt.imshow(Z_pred, cmap = 'gray')
plt.show()
# Subtract & show
S = Z - Z_pred
plt.figure(4)
plt.imshow(S, cmap = 'gray')
plt.show()
# In[ ]:TASK 5
n = np.linspace(0,99,100)
w = np.sqrt(0.3) * np.random.randn(100)
x = np.sin(2*np.pi*0.015*n) + w
plt.figure(5)
plt.plot(x, 'b-')
plt.show()
scores = [] 
frequencies = []
for f in np.linspace(0, 0.5, 1000):
    # Create vector e. Assume data is in x. 
    # another way to get 
    n = np.arange(100) 
    # <compute -2*pi*i*f*n. Imaginary unit is 1j> 
    z = -2*np.pi*1j*f*n
    e = np.exp(z)
    # <compute abs of dot product of x and e> 
    score = np.abs(x @ e)
    scores.append(score) 
    frequencies.append(f)
#the np.argmax(scores)  QAQ非常简洁  
fHat = frequencies[np.argmax(scores)]
print(fHat)
print('yes,result close to 0.015')

