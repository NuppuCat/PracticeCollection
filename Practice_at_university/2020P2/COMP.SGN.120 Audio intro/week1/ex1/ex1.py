# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:05:33 2020

@author: onepiece
"""
import  scipy
import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal as signal
import sounddevice as sd

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
# sd.play(y1, 6000)
plt.figure(1)
plt.plot(x1, y1,'r')
plt.xlim(0 , np.pi*2 )
plt.show()
# sd.play(y2, 6000)
plt.figure(2)
plt.plot(x2, y2,'g')
plt.xlim(0 , np.pi*2 )
plt.show()
# sd.play(y3, 6000)
plt.figure(3)
plt.plot(x3, y3,'b')
plt.xlim(0 , np.pi*2 )
plt.show()
# sd.play(y4, 6000)
plt.figure(4)
plt.plot(x4, y4,'y')
plt.xlim(0 , np.pi*2 )
plt.show()

y = y1+y2+y3+y4
# sd.play(y, 6000)
plt.figure(5)
plt.plot( x4,y,'y')
plt.xlim(0 , np.pi*2 )
plt.show()

##
fy = scipy.fft(y,512*2*2)
my = np.abs(fy)
py = np.angle(fy)

plt.figure(6)
plt.cla()
# plt.magnitude_spectrum(y, Fs=sample)
plt.plot(my)
plt.show()
plt.figure(7)
plt.cla()
# plt.phase_spectrum(y)
plt.plot(py)
plt.show()

print('nfft will limit the frequency of the plot.And for amplitude, the shape of it does not change, only become bigger')
print('While the phase will change alot')
# In[ ]:bonus problem
from scipy import signal
import wave
ry = signal.resample(y,int(len(y)/2))
# sd.play(ry,sample/2)
fry = scipy.fft(ry)
fy1 = scipy.fft(y)
plt.figure(12)
plt.cla()
# plt.magnitude_spectrum(ry, Fs=sample)
plt.plot(np.abs(fry))
plt.show()
plt.figure(13)
plt.cla()
# plt.magnitude_spectrum(ry, Fs=sample)
plt.plot(np.abs(fy1))
plt.show()

# In[ ]:
# import wave  
# f = wave.open('audio1.wav')
# y = f.readframes()
# print(y)
import soundfile as sf

data,fs = sf.read('audio1.wav', dtype='float32')
# sd.play(data, fs)
r =  range(int(fs/2),fs)
hdata = data[r]

plt.figure(8)
plt.cla()
plt.plot(hdata)
plt.show()
# start_from: the start time(from n second); dtime: how long after it;
def getnext(start_from,dtime,data,fs):
    a  = range(int(start_from*fs),int((start_from+dtime)*fs))
    s  = data[a]
    plt.figure(9)
    plt.cla()
    plt.plot(s)
    plt.show()
    fy = scipy.fft(s)
    plt.figure(10)
    plt.cla()
    plt.plot(np.abs(fy))
    plt.show()           

getnext(1,0.1,data,fs)

data2,fs2 = sf.read('audio2.wav', dtype='float32')
sd.play(data2, fs2)
hdata2 = data2[r]
plt.figure(11)
plt.cla()
plt.plot(hdata2)
plt.show()
getnext(1,0.1,data2,fs2)

print('e) the audio sound have more details, the sum of sinusoids is more simple')
