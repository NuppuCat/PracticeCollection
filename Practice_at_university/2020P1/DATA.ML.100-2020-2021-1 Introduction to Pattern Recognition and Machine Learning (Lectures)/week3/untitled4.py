# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:41:08 2020

@author: onepiece
"""

import numpy as np
import matplotlib.pyplot as plt
# Some random experiments with 2D Gaussians
mu1 = [165,60]
cov1 = [[10,0],[0,5]]
mu2 = [180,80]
cov2 = [[6,0],[0,10]]
x1 = np.random.multivariate_normal(mu1, cov1, 100)
x2 = np.random.multivariate_normal(mu2, cov2, 100)
plt.plot(x1[:,0],x1[:,1],'rx')
plt.plot(x2[:,0],x2[:,1],'gx')
