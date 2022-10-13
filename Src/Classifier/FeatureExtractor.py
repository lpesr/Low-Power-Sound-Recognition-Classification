from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.io
import numpy as np

def plot_fft(filename):
    samplerate, transform = wavfile.read(filename)

    plt.plot(np.fft.rfft(transform), label="Left channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Hz")
    plt.show()

def spectral_centroid(x, samplerate):
    magnitudes = np.abs(np.fft.rfft(x))                       # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes)      # return weighted mean

plot_fft('../../Data/5-251489-A-24.wav')
samplerate, data = wavfile.read('../../Data/5-251489-A-24.wav')
length = data.shape[0] / samplerate
miliseconds = int(samplerate / 1000)

datapoints = [data[i:i+(miliseconds * 10)] for i in range(0, len(data), (miliseconds * 10))]

centroids = []
for i in range(len(datapoints)):
    centroids.append(spectral_centroid(datapoints[i], samplerate))

#centroids = map(spectral_centroid, datapoints, samplerate)

plt.plot(centroids, label="Left channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Hz")
plt.show()