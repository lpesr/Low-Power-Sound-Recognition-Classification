import FeatureExtractor as fe
import numpy as np
import matplotlib.pyplot as plt
import math
import librosa
from scipy.io import wavfile

frameSize = 500

#plot_fft('../../Data/5-251489-A-24.wav')

centroids = fe.wav_to_spectral_centroid('../../Data/5-251489-A-24.wav', frameSize)
centroidsTwo = fe.wav_to_spectral_centroid('../../Data/5-263902-A-36.wav', frameSize)
centroidsThree = fe.wav_to_spectral_centroid('../../Data/crying_baby/2-66637-B-20.wav', frameSize)

testWav, sr = librosa.load('../../Data/5-251489-A-24.wav')
test = librosa.feature.spectral_centroid(y=testWav, sr=sr, n_fft=frameSize, hop_length=500)[0]

plt.plot(centroids, color='r')
plt.plot(centroidsTwo, color='b')
plt.plot(centroidsThree, color='g')
plt.plot
plt.legend()
plt.xlabel("Centroids")
plt.ylabel("Spectral Rate")
plt.show()
