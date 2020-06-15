import matplotlib.pyplot as plt
import random
import numpy as np

from scipy import signal

import scipy.io.wavfile as wav

nperseg = 1024
window = 'triang'
phase1 = 2 * np.pi * random.random() #initialize random phase

print("\n Starting phase: ", phase1, "\n")
fs, audio = wav.read("inception_sound_track.wav")
audiosfft = audio[:, 0] #retrieve a single channel


f, _, audiosfft = signal.stft(audiosfft, fs=fs, window=window, noverlap = nperseg // 2, nperseg=nperseg)


phase = np.random.uniform(0,2*np.pi,audiosfft.shape) #np.pi /2 #initialize phase matrix to be the same shape as audiosfft matrix

print("Processing...")
orig_mag = abs(audiosfft) #obtain magnitude
orig_phase = np.angle(audiosfft)
for iter in range(50):

    audiosfft = np.multiply(orig_mag, np.exp(np.multiply(1j, phase)))

    _, newsignal = signal.istft(audiosfft, fs)
    newsignal = newsignal.astype(complex).real

    f, _, audiosfft = signal.stft(newsignal, fs=fs, window=window, noverlap = nperseg // 2, nperseg=nperseg)
    phase = np.angle(audiosfft)

#wav.write('just_for_kicks.wav',fs,newsignal.astype('int16'))
print("\n \nWritten successfully")
print(orig_phase.shape)
print(phase.shape)
A = np.subtract(orig_phase, phase)
A_norm = np.linalg.norm(A, 'fro')
print(A_norm.shape)
A_mag = np.sqrt(np.sum(np.square(A_norm)))
print("\nThe Forbenius norm of the difference matrix is: ", A_mag)
