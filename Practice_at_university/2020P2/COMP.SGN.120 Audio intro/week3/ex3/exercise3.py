import numpy as np
from scipy.io import wavfile
from numpy.fft import fft, ifft, fftshift
import librosa as lb
#import sys
from scipy.signal import hann



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
    
    
# A Loop for overap add reconstruction  with no spectral processing in between    
audioIn, fs=lb.load('audio.wav', sr=None)   # read audio

audioOut = np.zeros(len(audioIn))        # placeholder for reconstructed audio
wLen = int(0.032*fs)                   # window length
winAn = np.sqrt(hann(wLen, sym=False)) # analysis window
winSyn =winAn


winHopAn = int(0.008*fs)             # Hop length or frame advance
inInd = 0

while inInd< len(audioIn)-wLen:

  # selct the frame and multiply with window function
  frame = audioIn[inInd:inInd+wLen]* winAn 
  
  # compute DFT
  f = fft(frame)
  
  # save magnitudes and phases
  mag_f = np.abs(f)
  phi0 = np.angle(f) 
  
  ####################
  # processing in spectral domain 
  #######################
  
  # Recover the complex FFT back
  ft = (abs(f)* np.exp(1j*phi0))  
  
  # inverse DFT and windowing
  frame = np.real(ifft(ft))*winSyn
  
  # Ovelap add
  audioOut[inInd :inInd +wLen] =  audioOut[inInd :inInd +wLen] + frame
  
  # frame advance by winHopAn
  inInd = inInd + winHopAn
  


