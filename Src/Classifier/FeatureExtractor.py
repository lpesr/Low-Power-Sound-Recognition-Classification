import numpy as np
import matplotlib.pyplot as plt
import math
import librosa
from scipy.io import wavfile

def plot_fft(filename):                                                                     # plot the frequencies of a wav file
    samplerate, transform = wavfile.read(filename)

    plt.plot(np.fft.rfft(transform))
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Hz")
    plt.show()

def calculate_spectral_centroid(data, sampleRate):
    magnitudes = np.abs(np.fft.rfft(data))                                                  # magnitudes of positive frequencies
    length = len(data)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/sampleRate)[:length//2+1])                    # positive frequencies
    sums = np.sum(magnitudes)
    return 0 if sums == 0 else np.sum(magnitudes*freqs) / sums                                    # return weighted mean

def wav_to_spectral_centroid(fileName, frameSize):
    sampleRate, spectralDencity = wavfile.read(fileName)
    length = spectralDencity.shape[0] / sampleRate

    frames = [spectralDencity[i:i+(frameSize)] for i in range(0, len(spectralDencity), (frameSize))]   # group spectralDencity into frames
    return [calculate_spectral_centroid(frame, sampleRate) for frame in frames]             # return list of spectral centroids

frameSize = 500

#plot_fft('../../Data/5-251489-A-24.wav')

centroids = wav_to_spectral_centroid('../../Data/5-251489-A-24.wav', frameSize)
centroidsTwo = wav_to_spectral_centroid('../../Data/5-263902-A-36.wav', frameSize)
centroidsThree = wav_to_spectral_centroid('../../Data/crying_baby/2-66637-B-20.wav', frameSize)

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
