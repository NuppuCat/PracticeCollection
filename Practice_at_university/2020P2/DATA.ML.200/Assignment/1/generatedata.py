# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:01:22 2020

@author: onepiece
"""

import cv2
import numpy as np
from matplotlib.image import imread

array_of_img=[]

def getextenddata(img,a,dirname):
    
    
    # 沿着横纵轴放大2倍，然后平移(-150,-240)，最后沿原图大小截取，等效于裁剪并放大
    M_crop_trans = np.array([
        [1.5, 0, -6],
        [0, 1, 0]
    ], dtype=np.float32)
    
    img_crop_trans = cv2.warpAffine(img, M_crop_trans, (28, 28))
    a1 = str(a+6000)
    cv2.imwrite(dirname+'/'+a1+'.jpg', img_crop_trans)
    
    # x轴的剪切shear变换，角度45°
    theta = 20 * np.pi / 180
    M_shear = np.array([
        [0.5, np.tan(theta), -6],
        [0, 1, 0]
    ], dtype=np.float32)
    
    img_sheared = cv2.warpAffine(img, M_shear, (28, 28))
    a2 = str(a+6000*2)
    cv2.imwrite(dirname+'/'+a2+'.jpg', img_sheared)
    M_shear2 = np.array([
        [1, 0, 0],
        [np.tan(theta), 0.5, 6]
    ], dtype=np.float32)
    
    img_sheared2 = cv2.warpAffine(img, M_shear2, (28, 28))
    a3 =str( a+6000*3)
    cv2.imwrite(dirname+'/'+a3+'.jpg', img_sheared2)
    
    # 顺时针旋转，角度45°
    M_rotate = np.array([
        [np.cos(theta), -np.sin(theta), 10],
        [np.sin(theta), np.cos(theta), -3]
    ], dtype=np.float32)
    
    img_rotated = cv2.warpAffine(img, M_rotate, (28, 28))
    a4 = str(a+6000*4)
    cv2.imwrite(dirname+'/'+a4+'.jpg', img_rotated)
    # 沿着横纵轴放大2倍，然后平移(-150,-240)，最后沿原图大小截取，等效于裁剪并放大
    M = np.array([
        [1, 0, 0],
        [0, 1.5, -6]
    ], dtype=np.float32)
    
    img_transformed = cv2.warpAffine(img, M, (28, 28))
    a5 = str(a+6000*5)
    cv2.imwrite(dirname+'/'+a5+'.jpg', img_transformed)
    
    # # 某种变换，具体旋转+缩放+旋转组合可以通过SVD分解理解
    # M = np.array([
    #     [1, 1.5, -400],
    #     [0.5, 2, -100]
    # ], dtype=np.float32)
    
    # img_transformed = cv2.warpAffine(img, M, (28, 28))
    # cv2.imwrite('img_transformed.jpg', img_transformed)

import os

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        # print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        #print(img)
       
# array_of_img = np.array(array_of_img)

direct='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/1/train/train/10'
# direct='D:/GRAM/MasterProgramme/Tampere/DATA.ML.200/Assignment/1/t'
read_directory(direct)
# for i in range(0,len(array_of_img)):
#     a = i+1;
    
#     getextenddata(array_of_img[i],a,direct)
    
    
    
    
    
    
    
