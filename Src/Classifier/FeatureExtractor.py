import numpy as np
import matplotlib.pyplot as plt
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
    return np.sum(magnitudes*freqs) / np.sum(magnitudes)                                    # return weighted mean

def wav_to_spectral_centroid(fileName, frameSize):
    sampleRate, spectralDencity = wavfile.read(fileName)
    length = spectralDencity.shape[0] / sampleRate

    frames = [spectralDencity[i:i+(frameSize)] for i in range(0, len(spectralDencity), (frameSize))]   # group spectralDencity into frames
    return [calculate_spectral_centroid(frame, sampleRate) for frame in frames]             # return list of spectral centroids

frameSize = 500

centroids = wav_to_spectral_centroid('../../Data/1-59513-A.ogg', frameSize)

testWav, sr = librosa.load('../../Data/1-59513-A.ogg')
test = librosa.feature.spectral_centroid(y=testWav, sr=sr, n_fft=frameSize, hop_length=500)[0]

plt.plot(centroids, color='r')
plt.plot(test, color='b')
plt.plot
plt.legend()
plt.xlabel("Centroids")
plt.ylabel("Spectral Rate")
plt.show()
