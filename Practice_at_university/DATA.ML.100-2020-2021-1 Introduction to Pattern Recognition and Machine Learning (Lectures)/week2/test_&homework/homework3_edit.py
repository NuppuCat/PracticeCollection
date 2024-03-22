#!/usr/bin/env python
# coding: utf-8

# # Probability distributions
# 
# At the core of statistical methods inc. Bayesian inference is modelling the underlying phenomenon. By modeling we start to understand our measurements and can develop techniques to maximise information, e.g., by reducing noise in digital camera sensor or microphone.
# 
# ![image.png](attachment:image.png)

# In[1]:


# Let's have a bayesien look on the male/female classification problem
import matplotlib.pyplot as plt
import numpy as np
plt.xlabel('x: height [cm]')
plt.axis([140,200,-0.25,0.5])
                 #     均值 标准差 个数  
x_1 = np.random.normal(165,5,20) # Measurements from the class 1
x_2 = np.random.normal(180,6,20) # Measurements from the class 2
plt.plot(x_1,np.zeros(len(x_1)),'rx')
plt.plot(x_2,np.zeros(len(x_2)),'ko')
plt.show()


# In[2]:


# Let's estimate the probability densities using Gaussian kernels
import scipy.stats as stats
x = np.arange(100,200,1)


# For comparison let's also print Gaussians
mu1 = np.mean(x_1)
mu2 = np.mean(x_2)
sigma1 = np.std(x_1)
sigma2 = np.std(x_2)
x = np.arange(100,200,1)
plt.plot(x, stats.norm.pdf(x, mu1, sigma1),'r--')
plt.plot(x, stats.norm.pdf(x, mu2, sigma2),'g--')
yval1guassian = stats.norm.pdf(x, mu1, sigma1)
yval2guassian = stats.norm.pdf(x, mu2, sigma2)

yval1 = np.zeros(len(x))
yval2 = np.zeros(len(x))
#set a range of kernel_width
rangeOfKW = np.arange(0.1,10,0.1)
cost = np.zeros(len(rangeOfKW))
i = 0
#try every kernel_width
for kernel_width in rangeOfKW:
    for xind, xval in enumerate(x):
        yval1[xind] = sum(stats.norm.pdf(x_1, xval, kernel_width))
    for xind, xval in enumerate(x):
        yval2[xind] = sum(stats.norm.pdf(x_2, xval, kernel_width))
    cha1 = yval1/sum(yval1)-yval1guassian
    cha2 = yval2/sum(yval2)-yval2guassian
    #save the differences of the gaussians distribution and the kernel distribution
    cost[i] = sum(np.power(cha1,2))+sum(np.power(cha2,2))
    yval1 = np.zeros(len(x))
    yval2 = np.zeros(len(x))
    i = i+1
#find  a value that makes the kernel distribution to match the Gaussian distributions as well as possible.    
idx = np.where(cost==np.min(cost))
kernel_width_best = rangeOfKW[idx]
print(f"best kernel_width is {kernel_width_best}")

# calculat best val to plot
for xind, xval in enumerate(x):
    #                                    点   均值/中心点  方差
        yval1[xind] = sum(stats.norm.pdf(x_1, xval, kernel_width_best))
for xind, xval in enumerate(x):
        yval2[xind] = sum(stats.norm.pdf(x_2, xval, kernel_width_best))
# We normalize values to sum one (this is ad hoc)
plt.plot(x, yval1/sum(yval1),'r-')
plt.plot(x, yval2/sum(yval2),'g-')




# In[3]:


# Some random experiments with 2D Gaussians
mu1 = [165,60]
cov1 = [[10,0],[0,5]]
mu2 = [180,80]
cov2 = [[6,0],[0,10]]
x1 = np.random.multivariate_normal(mu1, cov1, 100)
x2 = np.random.multivariate_normal(mu2, cov2, 100)
plt.plot(x1[:,0],x1[:,1],'rx')
plt.plot(x2[:,0],x2[:,1],'gx')






# ## References
# 
# C.M. Bishop (2006): Pattern Recognition and Machine Learning, Chapter 1-2.
