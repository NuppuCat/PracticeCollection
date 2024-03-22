#!/usr/bin/env python
# coding: utf-8

# # SGN-14007 Introduction to Audio Processing 
# ##                Exercise 5

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,dct
import librosa
from scipy import signal
import librosa.display
import soundfile as sf


# ### Problem 1

# In[3]:


#############problem 1############## 
s,sr=sf.read('audio.wav')
n_fft=512
win_size = 512
hop_size=128
n_mel = 40
fbank = librosa.filters.mel(sr,n_fft,n_mel)
########## plot filterbanks #########################
plt.figure(figsize=(10, 4))
librosa.display.specshow(fbank,x_axis='linear')
plt.ylabel('Mel filter')
plt.colorbar()
plt.title('filterbanks')
plt.tight_layout()
plt.show()


# ### As shown in the figure above, with the same interval in mel scale, it covers larger range for high frequencies than low frequencies in linear scale hz 

# ### Problem 2

# In[4]:


#############problem 2#######################

pre_emphasis = 0.97
s_ = np.append(s[0], s[1:] - pre_emphasis * s[:-1])

window = signal.hamming(win_size,sym= False)

n_frames=int((len(s_)-win_size)/hop_size)+1
mfccs = np.zeros((40,n_frames),dtype=np.float32)
power_spectrums = np.zeros((n_fft//2+1,n_frames),dtype=np.float32)
mel_spectrums = np.zeros((40,n_frames),dtype=np.float32)
log_mel_spectrums = np.zeros((40,n_frames),dtype=np.float32)
for i in np.arange(0,n_frames):   
    s_seg = s_[i*hop_size:i*hop_size+win_size]        
    s_win_seg = window * s_seg      
    spectrum  = fft(s_win_seg,n_fft)  
    spectrum=spectrum[:n_fft//2+1] 
    power_spectrum = (np.power(np.abs(spectrum),2)).reshape((n_fft//2+1,1))
    power_spectrums[:,i] = power_spectrum.flatten()
    mel_spectrum=np.dot(fbank,power_spectrum)
    log_mel_spectrum = 20 * np.log10(mel_spectrum)
    mfcc = dct(log_mel_spectrum, axis=0,norm='ortho')[:40]
    mel_spectrums[:,i] = mel_spectrum.flatten()
    log_mel_spectrums[:,i]= log_mel_spectrum.flatten()
    mfccs[:,i] = mfcc.flatten()


# 

# In[6]:


########## plot power_spectrums #########################
plt.figure(figsize=(10, 4))
librosa.display.specshow(20*np.log10(power_spectrums), x_axis='time',y_axis='linear',sr=sr, hop_length=hop_size)
plt.colorbar()
plt.title('log_power_spectrums')
plt.tight_layout()
plt.show()


########## plot mel_spectrums #########################
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrums, x_axis='time',sr=sr, hop_length=hop_size)
plt.colorbar()
plt.title('mel_spectrums')
plt.tight_layout()
plt.show()
########## plot log_mel_spectrums #########################
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrums, x_axis='time',sr=sr, hop_length=hop_size)
plt.colorbar()
plt.title('log_mel_spectrums')
plt.tight_layout()
plt.show()

########## plot own mfcc #########################
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs,cmap='Spectral', x_axis='time',sr=sr, hop_length=hop_size)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

############# plot mfcc using librosa #####################
mfcc_librosa = librosa.feature.mfcc(y=s, sr=sr, S=None, n_mfcc=40, norm='ortho',n_fft =n_fft,hop_length = hop_size )
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_librosa,cmap='Spectral', x_axis='time',sr=sr, hop_length=hop_size)
plt.colorbar()
plt.title('MFCC using librosa')
plt.tight_layout()
plt.show()


# ### From the last two figures, it can be seen that there is not much difference between the MFCC implemented by my own and by librosa

# ### Bonus

# In[8]:


################bonus point ################
def Mel_filterbank(fs,n_mel,n_fft):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mel + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((n_fft + 1) * hz_points / fs)
    
    fbank = np.zeros((n_mel, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mel + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    enorm = 2.0 / (hz_points[2:n_mel+2] - hz_points[:n_mel])
    fbank *= enorm[:, np.newaxis]
    return fbank


# In[10]:


own_fbank = Mel_filterbank(sr,n_mel,n_fft)

plt.figure(figsize=(10, 4))
librosa.display.specshow(own_fbank, x_axis='linear')
plt.ylabel('Mel filter')
plt.colorbar()

plt.title('own filterbanks')
plt.tight_layout()
plt.show()


# ### By comparing the first figure with the last figure, there is not much difference.

# ### You can aslo plot the mel filterbank in a row-wise

# In[11]:


plt.figure(figsize=(10,4))
for i in np.arange(own_fbank.shape[0]):
    plt.plot(own_fbank[i,:])


# In[ ]:




