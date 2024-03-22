#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,dct
import librosa
from scipy import signal
import librosa.display
import soundfile as sf
import sounddevice as sd


# ### Question 1

# In[2]:


def plot(audio, sr,name):
    
    stft = librosa.core.stft(audio)
    cqt = librosa.core.cqt(audio,sr = sr)
    chroma_stft = librosa.feature.chroma_stft(audio,sr)

    plt.figure(figsize=(10,4))
    librosa.display.specshow(librosa.amplitude_to_db(stft,ref=np.max), y_axis='linear', x_axis='time',cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('power spectrogram of '+name)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(10,4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(cqt), ref=np.max),sr=sr, x_axis='time', y_axis='cqt_note',cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum '+name)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time',cmap='coolwarm')
    plt.colorbar()
    plt.title('Chromagram '+name)
    plt.tight_layout()
    plt.show()


# In[3]:


brahms,sr = sf.read('brahms_hungarian_dance_5_short.wav')
classic,sr = sf.read('classic_rock_beat.wav')
conga,sr = sf.read('conga_groove.wav')
latin,sr = sf.read('latin_groove_short.wav')

plot(brahms,sr,'brahms')
plot(classic,sr,'classic')
plot(conga,sr,'conga')
plot(latin,sr,'latin')


# ### In the spectrogram, it is quite hard to see the features due to the redundant information. However, it is easier to see from the rest of features representations.

# ### Question 2

# In[8]:


hop_length = 512
onset_envelope = librosa.onset.onset_strength(classic, sr=sr, hop_length=hop_length)
onset_frames = librosa.util.peak_pick(onset_envelope,pre_max = 1, post_max=1, pre_avg=1, post_avg=1, delta=0.5, wait=1)
onset_times = librosa.frames_to_time(onset_frames)
S = librosa.stft(classic)
logS = librosa.amplitude_to_db(abs(S))


# In[12]:


plt.figure(figsize=(10,4))
librosa.display.waveplot(classic,sr)
plt.vlines(onset_times,ymax=1,ymin=-1,colors='r')

plt.figure(figsize=(10,4))
librosa.display.specshow(logS,x_axis='time',y_axis='log')
plt.vlines(onset_times,ymax=sr/2,ymin=-1,colors='r')


clicks = librosa.clicks(frames=onset_frames, sr=sr, hop_length=hop_length, length=len(classic))
clicked_audio_stero =np.array(np.vstack([classic, clicks])).T
clicked_audio_mono = classic + clicks
#sd.play(classic,sr)
#sd.play(clicked_audio_mono,sr)
#sd.play(clicked_audio_stero,sr)


# #### If delta=3 or even higher, some peaks are not picked up. If we set delta=0.5, most of the peaks can be detected.

# ### Bonus

# In[6]:


def relu(X):
   return np.maximum(0,X)

audio = classic
stft = librosa.core.stft(audio,hop_length= hop_length)
log_stft=librosa.amplitude_to_db(stft,ref=np.max)
log_com = np.log(1+1*np.abs(log_stft))
onset_envelope_own = relu(np.sum(np.diff(log_com),axis=0))
onset_frames = librosa.util.peak_pick(onset_envelope_own,pre_max = 1, post_max=1, pre_avg=1, post_avg=1, delta=1, wait=1)
onset_times = librosa.frames_to_time(onset_frames)

plt.figure(figsize=(10,4))
librosa.display.waveplot(classic,sr)
plt.vlines(onset_times,ymax=1,ymin=-1,colors='r')


# In[ ]:





# In[ ]:




