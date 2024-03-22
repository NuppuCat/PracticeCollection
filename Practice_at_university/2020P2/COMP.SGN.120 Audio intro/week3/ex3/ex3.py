# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:10:51 2020

@author: onepiece
"""
import  scipy
import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal as signal
import sounddevice as sd
import librosa as lb
from scipy.signal import hann
from numpy.fft import fft, ifft, fftshift
Fs = 16000
cf = 880
mf =220
A = 1
mi = 2
time = 1
t = np.linspace(0, cf * time * 2 * np.pi  , Fs*time)
t1 = np.linspace(0, mf * time * 2 * np.pi  , Fs*time)
y1 = mi*np.sin(t1)
y = A * np.sin(t+y1)
# sd.play(y, Fs)
# plt.figure(1)
# plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
# plt.plot(y[:500])
D = lb.stft(y)    
m2 = np.abs(D)
m2 = m2**2                


def plot_spectrogram (spec):
    plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(spec,origin='lower',aspect='auto')
    locs, labels = plt.xticks()
    locs_=[np.round((i/locs[-1]*len(y)/Fs),decimals=1) for i in locs]
    plt.xticks(locs[1:-1], locs_[1:-1])
    locs, labels = plt.yticks()
    locs_=[int((i/locs[-1]*Fs//2)) for i in locs]
    plt.yticks(locs[1:-1], locs_[1:-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Fre (Hz)')
    

# dft = fft(y)
# freq = np.linspace(0, Fs/2, len(y)//2)
# plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
# plt.plot(freq, abs(dft[:len(dft)//2]))

# In[ ]:


def princarg(phase_in):
  """
  Computes principle argument,  wraps phase to (-pi, pi]
  """
  phase = np.mod(phase_in + np.pi,-2*np.pi)+np.pi;
  return phase
  



def delta_phi_(Phase_current, Phase_previous, winHopAn, wLen):
    """
    Function for calculating unwrapped phase difference between consecutive frames
    
    Phase_current: current frame phase
    Phase_previous: previous frame phase
    winHopAn: Analysis hop length
    wLen: window length
    """
    
    # nominal phase increment for the analysis hop size for each bin
    omega = 2*np.pi*(winHopAn/wLen)*np.arange(0, wLen)
    delta_phi = omega + princarg(Phase_current-Phase_previous-omega)
    
    return delta_phi
    
R = 1.4   
# A Loop for overap add reconstruction  with no spectral processing in between    
audioIn, fs=lb.load('audio.wav', sr=None)   # read audio

audioOut = np.zeros(int(len(audioIn)*R))        # placeholder for reconstructed audio
wLen = int(0.032*fs)                   # window length
winAn = np.sqrt(hann(wLen, sym=False)) # analysis window
winSyn =winAn


winHopAn = int(0.008*fs)  
winHopSyn = round(winHopAn*R)           # Hop length or frame advance
inInd = 0
outInd=0
Phase_previous = np.zeros(wLen)
phase_req=np.zeros(wLen)
psi = np.zeros(wLen)
while inInd< len(audioIn)-wLen:

  # selct the frame and multiply with window function
  frame = audioIn[inInd:inInd+wLen]* winAn 
  
  # compute DFT
  f = fft(fftshift(frame)) 
  
  # save magnitudes and phases
  mag_f = np.abs(f)
  phi0 = np.angle(f) 
  Phase_current = phi0
  delta_phi = delta_phi_(Phase_current, Phase_previous, winHopAn, wLen)
  
  sy_phase = phase_req + R * delta_phi
  
  phase_req = princarg(sy_phase)
  
  """solution
  # selct the frame and multiply with window function
  ""
  frame = audioIn[inInd:inInd+wLen]* winAn
    
  # compute DFT
  f = fft(fftshift(frame))  
  #f = fft(frame)
  
  # phase processing in spectral domain
  #delta_phi= omega + princarg(np.angle(f)-phi0-omega)
  delta_phi = delta_phi_(np.angle(f), phi0, winHopAn, wLen) 
  psi = princarg(psi+delta_phi*R)

  # save  phase  for the next iteration
  phi0 = np.angle(f) 

  # Recover the complex FFT back
  ft = (abs(f)* np.exp(1j*psi)) 
  
  # inverse DFT and windowing
  #frame = np.real(ifft(ft))*winSyn;
  frame = fftshift(np.real(ifft(ft)))*winSyn
  
  # Ovelap add
  audioOut[outInd:outInd+wLen] =  audioOut[outInd:outInd+wLen] + frame;
  
  # frame advance
  inInd = inInd + winHopAn;
  outInd = outInd + winHopSyn;
  cnt=cnt+1
  """""
  ####################
  # processing in spectral domain 
  #######################
  
  # Recover the complex FFT back
  ft = (abs(f)* np.exp(1j*phase_req))  
  
  # inverse DFT and windowing
  frame = fftshift(np.real(ifft(ft)))*winSyn
  
  # Ovelap add
  # audioOut[outInd :outInd +wLen] =  audioOut[outInd :outInd +wLen] + frame
  audioOut[outInd:outInd+wLen] =  audioOut[outInd:outInd+wLen] + frame;
  # frame advance by winHopAn
  inInd = inInd + winHopAn
  outInd = outInd + winHopSyn;
  Phase_previous = Phase_current

print("This adjusts the phase difference between adjacent frames to what it must be for the modified hop size")
time_axis = np.linspace(0, len(audioIn)/fs, len(audioIn), endpoint=False)
# sd.play(audioOut, fs)
# sd.play(audioIn, fs)
plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(audioIn, 'b')
plt.plot(audioOut, 'r')
