import FeatureExtractor as fe
import numpy as np
import matplotlib.pyplot as plt
import math
import librosa
from scipy.io import wavfile

frameSize = 500

#plot_fft('../../Data/5-251489-A-24.wav')

centroids = fe.wav_to_spectral_centroid('../../Data/airplane/1-11687-A-47.wav', frameSize)
centroidsTwo = fe.wav_to_spectral_centroid('../../Data/dog/1-30226-A-0.wav', frameSize)
centroidsThree = fe.wav_to_spectral_centroid('../../Data/crying_baby/2-66637-B-20.wav', frameSize)

zeroCrossingRate = fe.wav_to_ZCR('../../Data/airplane/1-11687-A-47.wav', 2048)

testWav, sr = librosa.load('../../Data/airplane/1-11687-A-47.wav')
testCentroids = librosa.feature.spectral_centroid(y=testWav, sr=sr, n_fft=frameSize, hop_length=500)[0]
testZCR = librosa.feature.zero_crossing_rate(y=testWav)[0]

plt.plot(zeroCrossingRate, color='b')
plt.show()

plt.plot(testZCR, color='r')
#plt.plot(centroids, color='r')
#plt.plot(centroidsTwo, color='b')
#plt.plot(centroidsThree, color='g')
plt.legend()
plt.xlabel("Centroids")
plt.ylabel("Spectral Rate")
plt.show()