# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import xml.dom.minidom
from xml.dom import Node

path = 'D:/GRAM/coursera/4CNN/week3/flyy/dataset/SCUT_HEAD_Part_A/Annotations'


import os
import os.path
import xml.dom.minidom


files= os.listdir(path)  # 得到文件夹下所有文件名称
s = []
for xmlFile in files:  # 遍历文件夹
    if not os.path.isdir(xmlFile):  # 判断是否是文件夹,不是文件夹才打开
        print("---------------")
        print(xmlFile)
        # xml文件读取操作
        # 将获取的xml文件名送入到dom解析
        dom = xml.dom.minidom.parse(path+'/'+xmlFile)
        root = dom.documentElement
        listInfos = []
        for child in root.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                dictAttr = {}
                for key in child.attributes.keys():
                    attr = child.attributes[key]
                    dictAttr[attr.name] = attr.value
                listInfos.append({child.nodeName: dictAttr})







# def multiple_Gaussian_distribution(X,mu,sd):
#     f = []
#     for x in X:
#         f.append(1/(sd*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sd**2))) 
#     return f

# mu = np.asarray([1,3,5]).T
# sd = [[4,2,1],[2,5,2],[1,2,3]]
# sd = np.asanyarray(sd)
# x1 = np.asarray([2,2,2]).T
# x2 = np.asarray([1,4,3]).T
# x3 = np.asarray([1,1,5]).T
# X = [x1,x2,x3]
# f = multiple_Gaussian_distribution(X,mu,sd)
# print(f)
# print("check the result by scipy")
# from scipy import stats
# y = stats.norm(mu, sd).pdf(x1)
# print(y)
# y = stats.norm(mu, sd).pdf(x2)
# print(y)
# y = stats.norm(mu, sd).pdf(x3)
# print(y)

# a = '/asdasdasdas'
# print(a.startswith('/'))