import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_fft(filename) :                                                                     # plot the frequencies of a wav file
    samplerate, transform = wavfile.read(filename)

    plt.plot(np.fft.rfft(transform))
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Hz")
    plt.show()

def convert_to_single_band(audioData):
    if len(audioData.shape) > 1 :
        if audioData.shape[1] > 1 :
            audioData = audioData[:,0]
    return audioData

def padd_and_snip_feature(feature, sampleRate, paddingSize, frameSize):
    padding = round((paddingSize - len(feature) * float(frameSize) / float(sampleRate)) * float(sampleRate) / float(frameSize))
    if padding > 0:
        return feature + [0] * padding
    else:
        return feature[:round(paddingSize * (float(sampleRate) / float(frameSize)))]

def get_wav_data(fileName, frameTime):
    sampleRate, audioData = wavfile.read(fileName)
    frameSize = int(sampleRate * frameTime)
    audioData = convert_to_single_band(audioData)
    return (sampleRate, audioData, frameSize, audioData)

def calculate_spectral_centroid(data, sampleRate) :
    magnitudes = np.abs(np.fft.rfft(data))                                                  # magnitudes of positive frequencies
    length = len(data)                    
    freqs = np.abs(np.fft.fftfreq(length, 1.0/sampleRate)[:length//2+1])                    # positive frequencies
    sums = np.sum(magnitudes)
    return 0 if sums == 0 else np.sum(magnitudes*freqs) / sums                              # return weighted mean

def wav_to_spectral_centroid(fileName, frameTime, paddingSize = 10) :
    (sampleRate, audioData, frameSize, audioData) = get_wav_data(fileName, frameTime)

    frames = [audioData[i:i+(frameSize)] for i in range(0, len(audioData), (frameSize))]    # group audioData into frames
    centroids = [calculate_spectral_centroid(frame, sampleRate) for frame in frames]        # return list of spectral centroids

    return padd_and_snip_feature(centroids, sampleRate, paddingSize, frameSize)

def wav_to_ZCR(fileName, frameTime, paddingSize = 10):
    (sampleRate, audioData, frameSize, audioData) = get_wav_data(fileName, frameTime)

    zeroCrossings = np.nonzero(np.diff(audioData > 0))[0]
    zcr = [0] * int(len(audioData) / frameSize + 1)
    for point in zeroCrossings :
        zcr[int(point / frameSize)] += 1

    return padd_and_snip_feature(zcr, sampleRate, paddingSize, frameSize)

def wav_threshold_normalization(wav, threshold) :
    index = next(x[0] for x in enumerate(wav) if x[1] > threshold)
    return wav[index:] + [0] * index