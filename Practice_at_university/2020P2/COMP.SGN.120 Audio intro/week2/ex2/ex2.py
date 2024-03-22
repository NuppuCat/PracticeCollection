# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:50:52 2020

@author: onepiece
"""
import  scipy
import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal as signal
import sounddevice as sd
import soundfile as sf

data,fs = sf.read('audio1.wav', dtype='float32')
# sd.play(data, fs)

dtime = 16/1000


# start_from: the start time(from n second); dtime: how long after it;
def getnext(start_from,dtime,data,fs):
    a  = range(int(start_from*fs),int((dtime+start_from)*fs))
   
    h = np.hamming(int((dtime+start_from)*fs)-int(start_from*fs))
    s  = data[a]
    
    
    # print(np.shape(s))
    s  = h * s
    
    fy = scipy.fft(s,int(dtime*fs))
              
    return(np.abs(fy))
# fy=getnext(1,0.1,data,fs)

# a = range(0,int(t*1000-dtime),int(dtime/2))
# print(a)
def getspectrum(start,dtime,data,fs,i):
    m = []
    t = len(data)/fs
    # a  = np.linspace(start,dtime+start,int(fs*dtime))
    # print(a)
    while start + dtime < t:
        f = getnext(start,dtime,data,fs)
        m.append(f) 
        start = start + dtime/2
        
    m = np.transpose(np.array(m))
    plt.figure(i)
    plt.cla()
    plt.imshow(m)    
    lm = np.log(m)
    plt.figure(i+1)
    plt.cla()
    plt.imshow(lm) 
getspectrum(0,dtime,data,fs,1)
print('two figures high light distribution looks similar, the color of original one is darker ')

# In[ ]: Probelm 2
import librosa

D = librosa.stft(data,n_fft=int(dtime*fs),window='hamming')    
m2 = np.abs(D)
plt.figure(3)
plt.cla()
plt.imshow(m2) 
print('Differences are that the number of frames with stft will *2,while y axis is about /2. I think the high light distribution is same')
A = 1

time_s = 3
sample = 8000
#从0到fi * time_s * 2 * np.pi，生成 sample* time_s个点
x1 = np.linspace(0, 100 * time_s * 2 * np.pi , sample* time_s)
y1 = A * np.sin(x1)
x2 = np.linspace(0, 500 * time_s * 2 * np.pi , sample* time_s)
y2 = A * np.sin(x2)
x3 = np.linspace(0, 1500 * time_s * 2 * np.pi , sample* time_s)
y3 = A * np.sin(x3)
x4 = np.linspace(0, 2500 * time_s * 2 * np.pi , sample* time_s)
y4 = A * np.sin(x4)
y = y1+y2+y3+y4
dtimes = np.array([0.016,0.032,0.064,0.128])    
data2,fs2 = sf.read('audio2.wav', dtype='float32')    
for i in dtimes:
    getspectrum(0,i,data,fs,i*1000)
    getspectrum(0,i,data2,fs2,i*10000)
    getspectrum(0,i,y,sample,i*100000)

print('window size *2,the number of frames will /2, the length of y axis *2. I do not know which is best,maybe 128ms,it can avoid forward masking?')
    
    
    
    
    
    
    
    
    
    
    
