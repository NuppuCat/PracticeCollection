#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt 
from scipy.fftpack import fft,ifft
from scipy import signal
import sounddevice as sd
import soundfile as sf
import sys
import librosa


# In[62]:


# A function to plot signal
def plot_signal(s,i):
    plt.figure()
    plt.plot(time_axis[i*hop_size:(i*hop_size+win_size)],s)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
# A function to plot spectrogram   
def plot_spectrogram (spec):
    plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(spec,origin='lower',aspect='auto')
    locs, labels = plt.xticks()
    locs_=[np.round((i/locs[-1]*len(s)/sr),decimals=1) for i in locs]
    plt.xticks(locs[1:-1], locs_[1:-1])
    locs, labels = plt.yticks()
    locs_=[int((i/locs[-1]*sr//2)) for i in locs]
    plt.yticks(locs[1:-1], locs_[1:-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Fre (Hz)')
    
    
# A function to compute inverse DFT and overlap-add reconstruction (Bonus problem)
def ISTFT(spectrogram, win, winsize, hopsize):
 
 window_synthesis = win
 n_frames=spectrogram.shape[1]
 output = np.zeros((s.shape[0]))
 for n_f in np.arange(n_frames):
    a=spectrogram[:,n_f]
    b=np.conjugate(spectrogram[:,n_f][-2:0:-1])
    c=np.concatenate((a,b))

    out = ifft(c).real
    out_wnd = window_synthesis*out                         # multiplication by a synthesis window
    output[n_f*hopsize:n_f*hopsize+winsize]=output[n_f*hopsize:n_f*hopsize+winsize]+ out_wnd # overlap add

 return output


# ### Problem 1

# In[54]:


epsilon = sys.float_info.epsilon # small positive value to avoid zeros inside the log

# audio signals
audios =['audio1.wav','audio2.wav','synthetic_sig.wav']

# assign audio
audio_0 = audios[0]  # you can change the audio and run for different audio files

# read audio
s,sr = sf.read(audio_0)

time_axis = np.linspace(0, len(s)/sr, len(s), endpoint=False)
win_size = int(0.1*sr)                        # window size
window = signal.hamming(win_size,sym= False)  # window function
# 整除
hop_size = win_size//2                        # hop size
n_fft=win_size                                # number of DFT points
n_frames=int((len(s)-win_size)/hop_size)+1    # number of frames
print(n_frames)
power_spectrogram = np.zeros((n_fft//2+1,n_frames),dtype=np.float32)   # placeholder for power spectrogram

for i in np.arange(0,n_frames):   
    s_seg = s[i*hop_size:i*hop_size+win_size]      # audio frame  
    s_win_seg = window * s_seg                     # multiply by a window
    
    spectrum  = fft(s_win_seg,n_fft)               # DFT spectrum
    spectrum=spectrum[:n_fft//2+1] 
    
    # let's pick a frame and plot spectrum with and without windowing
    if i == 15:
        plt.figure( figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')        
        t =  np.linspace(0, int(sr/2), int(n_fft/2+1))
        
        # spectrumm of windowed signal
        plt.plot(t[:len(t)//8], abs(spectrum[:len(t)//8])) # plot only a part of it for better visibility
        
        # spectrum of signal before windowing
        spectrum_  = fft(s_seg,n_fft)  
        spectrum_=spectrum_[:n_fft//2+1]  
        
        plt.plot(t[:len(t)//8], abs(spectrum_[:len(t)//8]), 'r') # plot only a part of it for better visibility
        plt.legend(['DFT after windowing', 'DFTbefore windowing'], loc='upper left')
        plt.title('Fig 1: DFT of the signal frame before (blue) and after windowing')
        
    power_spectrum = np.power(np.abs(spectrum),2)
    power_spectrogram[:,i] = power_spectrum
    
    
  
plot_spectrogram(power_spectrogram)
plt.title('Fig 2: Power spectrogram')

plot_spectrogram(20*np.log10(power_spectrogram+epsilon))
plt.title('Fig 3: Logrithmic power spectrogram')


# #### 1 a) Effect of windowing
# 
# Windowing in time domain implies convolution in frequency domain. So the process of windowing by default would lead to changes in the spectral content under analysis. The reason we use a tapering window , e.g., Hann, Hammming, is to avoid discontinuities at the window end points and their good frequency domain charecteristics in comparison to the rectangular window. A rectangular window will cause larger spectral smearing which can be seen from Fig. 1. 
# 
# ###### some additional resources
# 
# [Effects-of-Windowing](http://www.dataphysics.com/downloads/technical/Effects-of-Windowing-on-the-Spectral-Content-of-a-Signal-by-P.-Wickramarachi.pdf)
# 
# [Windowing](https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf)

# #### 1 c) Logarithm plots
# 
# As shown above logarithm basically compresses the range of the input and allows you to spot even small frequency components better.

# ### Problem 2
# 

# In[55]:


# spectrogram using a library function
spectrogram_librosa = librosa.stft(s,n_fft,win_length=win_size,hop_length=hop_size)
plt.figure( figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')        
plot_spectrogram(np.abs(spectrogram_librosa)**2)
plt.title('Power spectrogram using librosa library')


# In[60]:


# choose audio
audio = audios[1]
s,sr = sf.read(audio)

# window lengths
window_length_ms = [16, 32, 64, 128]
window_length= [int(i*0.001*sr) for i in window_length_ms]
time_axis = np.linspace(0, len(s)/sr, len(s), endpoint=False)

for i in window_length:
    
    window = signal.hamming(i,sym= False)
    hop_size = i//2
    n_fft=i
    n_frames=int((len(s)-i)/hop_size)+1
    power_spectrogram = np.zeros((n_fft//2+1,n_frames),dtype=np.float32)
    for j in np.arange(0,n_frames):   
        s_seg = s[j*hop_size:j*hop_size+i]        
        s_win_seg = window * s_seg    
        spectrum  = fft(s_win_seg,n_fft)  
        spectrum=spectrum[:n_fft//2+1] 
        power_spectrum = np.power(np.abs(spectrum),2)
        power_spectrogram[:,j] = power_spectrum
    plot_spectrogram(20*np.log10(power_spectrogram+epsilon))
    plt.title('Logrithmic power spectrogram of '+audio+' with window size '+str(int(np.round((i/sr*1000),decimals=1)))+'ms')


# #### 2 c) Choice of the window size 
# 
# The choice of the window size fixes time and frequency resolution for the analysis. It is always a trade-off. A large window size gives you a good frequency resolution but poor time resolution and vice-versa. For example , if your signal has frequency components which are close together you should use a larger window size in order to be able to resolve them.
# 
# In the above example for speech the frequecny components are best visible for window size 32 ms. For larger size of windows the stationarity assumption may not hold true as well. However, for music signals  you may need higher window lengths in order to resolve tones closely spaced in frequency (i.e., a better frequency resolution).

# ### Bonus problem 

# In[87]:


# Let us take a sinusoid with a low sampling rate for better visualization

sr = 500  #sampling rate
t = np.linspace(0, 1, 3*sr, endpoint=False)
s = np.sin(2*np.pi*t)

time_axis = np.linspace(0, len(s)/sr, len(s), endpoint=False)
win_size = int(0.1*sr) 


# Window functions , square root is taken as the window is multiplied two times once in 
# analysis and once in synthesis.

win1 = np.sqrt(signal.hamming(win_size,sym = True)) # window 1 : A symmetrical Hamming window
win2 = np.sqrt(signal.hamming(win_size,sym = False)) # window 2 : A periodic Hamming window
win3 = np.sqrt(signal.hann(win_size,sym = True)) # window 3 : A symmetrical Hann window
win4 = np.sqrt(signal.hann(win_size,sym = False)) # window 4 : A  periodic Hann window


window_func = [win1, win2, win3, win4]
plot_titles = ['Fig 3 a) Symmetrical Hamming window', 'Fig 3 b) Periodic Hamming window', 
               'Fig 3 c) Symmetrical Hann window', ' Fig 3 d) Periodic Hann window']

for window, title in zip(window_func, plot_titles):
    hop_size = win_size//2
    n_fft=win_size
    n_frames=int((len(s)-win_size)/hop_size)+1
    spectrogram = np.zeros((n_fft//2+1,n_frames),dtype=np.complex_)
    for i in np.arange(0,n_frames):   
        s_seg = s[i*hop_size:i*hop_size+win_size]        
        s_win_seg = window * s_seg

        spectrum  = fft(s_win_seg,n_fft)  
        spectrum=spectrum[:n_fft//2+1] 
        spectrogram[:,i] = spectrum


    s_recon = ISTFT(spectrogram, window, win_size, hop_size)


    plt.figure( figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')                           
    plt.plot(s, label = 'original_signal')
    plt.plot(s_recon,label = 'ŕeconstructed_signal')
    plt.legend(['original_signal', 'ŕeconstructed_signal'], loc = 'upper right' )
    plt.title(title)


# #### Constant overlap add (COLA) condition
# 
# In any analysis-synthesis processing with audio , the choice of window function is very important. If your window satisies the constant overlap add (COLA) condition, only then you will get the perfect reconstruction of your signal provided  no spectral processing is involved between the analysis and synthesis stages. The COLA condition depends upon the window function as well as the chosen overlap between the windows. 
# 
# As you can see in Fig 3, only the periodic versions of windows are COLA windows. In the next section let us see the overlapped sum of window functions alone to see what is actually happeining.

# In[86]:


# Let us see the overlap add of window functions alone. For this we will take the signal as 1.


sr = 500  #sampling rate
s = np.ones(sr* 3)   # unit amplitude signal

win_size = int(0.1*sr) 


# Window functions, square root is taken as the window is multiplied two times once in 
# analysis and once in synthesis.

win1 = np.sqrt(signal.hamming(win_size,sym = True)) # window 1 : A symmetrical Hamming window
win2 = np.sqrt(signal.hamming(win_size,sym = False)) # window 2 : A periodic Hamming window
win3 = np.sqrt(signal.hann(win_size,sym = True)) # window 3 : A symmetrical Hann window
win4 = np.sqrt(signal.hann(win_size,sym = False)) # window 4 : A  periodic Hann window


window_func = [win1, win2, win3, win4]
plot_titles = ['Fig 4 a) Symmetrical Hamming window', 'Fig 4 b) Periodic Hamming window', 
               'Fig 4 c) Symmetrical Hann window', ' Fig 4 d) Periodic Hann window']

for window, title in zip(window_func, plot_titles):
    hop_size = win_size//2
    n_fft=win_size
    n_frames=int((len(s)-win_size)/hop_size)+1
    spectrogram = np.zeros((n_fft//2+1,n_frames),dtype=np.complex_)
    for i in np.arange(0,n_frames):   
        s_seg = s[i*hop_size:i*hop_size+win_size]        
        s_win_seg = window * s_seg

        spectrum  = fft(s_win_seg,n_fft)  
        spectrum=spectrum[:n_fft//2+1] 
        spectrogram[:,i] = spectrum


    s_recon = ISTFT(spectrogram, window, win_size, hop_size)


    plt.figure( figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')                           
    
    plt.plot(s_recon,label = 'ŕeconstructed_signal')
    plt.legend(['sum of overlapped windows'], loc = 'lower right' )
    plt.title(title)


# You can check the COLA condition with _scipy.signal.check_cola_ function

# In[93]:


sr = 500  #sampling rate
s = np.ones(sr* 3)   # unit amplitude signal

win_size = int(0.1*sr) 


# Window functions, square root is taken as the window is multiplied two times once in 
# analysis and once in synthesis.

win1 = signal.hamming(win_size,sym = True)    # window 1 : A symmetrical Hamming window
win2 = signal.hamming(win_size,sym = False)   # window 2 : A periodic Hamming window
win3 = signal.hann(win_size,sym = True)       # window 3 : A symmetrical Hann window
win4 = signal.hann(win_size,sym = False)      # window 4 : A  periodic Hann window


window_func = [win1, win2, win3, win4]
titles = ['Symmetrical Hamming window', 'Periodic Hamming window', 
               'Symmetrical Hann window', 'Periodic Hann window']


for window, title in zip(window_func, titles):
    print(title + ' is COLA complaint :',  signal.check_COLA(window, win_size, win_size//2))


# #### Please note that COLA compliance depends on the overlap as well. In the above we have tested for 50 % overlap only.
