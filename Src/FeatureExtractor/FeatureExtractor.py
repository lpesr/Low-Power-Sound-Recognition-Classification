from cmath import log10
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import wavfile
import librosa

def plot_fft(filename) :                                                                     # plot the frequencies of a wav file
    samplerate, transform = wavfile.read(filename)

    plt.plot(np.fft.rfft(transform))
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Hz")
    plt.show()

def calculate_spectral_centroid(data, sampleRate) :
    magnitudes = np.abs(np.fft.rfft(data))                                                  # magnitudes of positive frequencies
    length = len(data)                    
    freqs = np.abs(np.fft.fftfreq(length, 1.0/sampleRate)[:length//2+1])                    # positive frequencies
    melFreqs = list(map(lambda f: 2595 * np.log10(1 + (f / 700)), freqs))
    sums = np.sum(magnitudes)
    return 0 if sums == 0 else np.sum(magnitudes*melFreqs) / sums                              # return weighted mean

def wav_to_spectral_centroid(fileName, frameTime, paddingSize = 10) :
    sampleRate, spectralDencity = wavfile.read(fileName)
    frameSize = int(sampleRate * frameTime)

    if len(spectralDencity.shape) > 1 :
        if spectralDencity.shape[1] > 1 :
            spectralDencity = spectralDencity[:,0]

    frames = [spectralDencity[i:i+(frameSize)] for i in range(0, len(spectralDencity), (frameSize))]   # group spectralDencity into frames
    centroids = [calculate_spectral_centroid(frame, sampleRate) for frame in frames]                   # return list of spectral centroids
    padding = round((paddingSize - len(centroids) * float(frameSize) / float(sampleRate)) * float(sampleRate) / float(frameSize))
    if padding > 0:
        return centroids + [0] * padding
    else:
        return centroids[:round(paddingSize * (float(sampleRate) / float(frameSize)))]

def wav_to_ZCR(fileName, frameTime, paddingSize = 10) :
    sampleRate, spectralDencity = wavfile.read(fileName)
    frameSize = int(sampleRate * frameTime)

    if len(spectralDencity.shape) > 1 :
        if spectralDencity.shape[1] > 1 :
            spectralDencity = spectralDencity[:,0]

    zeroCrossings = np.nonzero(np.diff(spectralDencity > 0))[0]
    zcr = [0] * int(len(spectralDencity) / frameSize + 1)
    for point in zeroCrossings :
        zcr[int(point / frameSize)] += 1

    padding = round((paddingSize - len(zcr) * float(frameSize) / float(sampleRate)) * float(sampleRate) / float(frameSize)) 
    if padding > 0:
        return zcr + [0] * padding
    else:
        return zcr[:round(paddingSize * (float(sampleRate) / float(frameSize)))]

def wav_threshold_normalization(wav, threshold) :
    index = next(x[0] for x in enumerate(wav) if x[1] > threshold)
    return wav[index:] + [0] * index