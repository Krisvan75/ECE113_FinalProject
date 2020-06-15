import matplotlib.pyplot as plt
import random
import numpy as np
import math
import cmath
import scipy.signal
import scipy.fft 
from scipy import fftpack
import scipy.io.wavfile as wav

phase1 = 2 * np.pi * random.random() #initialize random phase

print("\n Starting phase: ", phase1, "\n")
fs, audio = wav.read("inception_sound_track.wav")
audiofft = audio[:, 0] #retrieve a single channel

#print("\nThe original audio is", audio, "\n")

audiofft = scipy.fft.fft(audiofft)

phase = np.ones(audiofft.size) * phase1 #np.pi /2 

print("Processing...")
orig_mag = abs(audiofft) #obtain magnitude

for iter in range(100):
    audiofft = np.multiply(orig_mag, np.exp(np.multiply(1j, phase)))
    newsignal = scipy.fft.ifft(audiofft)
    newsignal = newsignal.astype(complex).real

    audiofft = scipy.fft.fft(newsignal)
    phase = np.angle(audiofft)

wav.write('output2.wav',fs,newsignal.astype('int16'))
print("\n \nWritten successfully")
