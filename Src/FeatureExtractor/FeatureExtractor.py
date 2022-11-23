import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa as lb

def plot_fft(filename):
    _, transform = wavfile.read(filename)

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
    return (sampleRate, audioData, frameSize)

def calculate_spectral_centroid(data, sampleRate):
    magnitudes = np.abs(np.fft.rfft(data))                                                  # magnitudes of positive frequencies
    length = len(data)                    
    freqs = np.abs(np.fft.fftfreq(length, 1.0/sampleRate)[:length//2+1])                    # positive frequencies
    sums = np.sum(magnitudes)
    return 0 if sums == 0 else np.sum(magnitudes*freqs) / sums                              # return weighted mean

def wav_to_spectral_centroid(fileName, frameTime, paddingSize = 10):
    (sampleRate, audioData, frameSize) = get_wav_data(fileName, frameTime)

    frames = [audioData[i:i+(frameSize)] for i in range(0, len(audioData), (frameSize))]    # group audioData into frames
    centroids = [calculate_spectral_centroid(frame, sampleRate) for frame in frames]        # return list of spectral centroids

    return padd_and_snip_feature(centroids, sampleRate, paddingSize, frameSize)

def wav_to_ZCR(fileName, frameTime, paddingSize = 10):
    (sampleRate, audioData, frameSize) = get_wav_data(fileName, frameTime)

    zeroCrossings = np.nonzero(np.diff(audioData > 0))[0]
    zcr = [0] * int(len(audioData) / frameSize + 1)
    for point in zeroCrossings :
        zcr[int(point / frameSize)] += 1

    return padd_and_snip_feature(zcr, sampleRate, paddingSize, frameSize)

def wav_threshold_normalization(wav, threshold):
    index = next(x[0] for x in enumerate(wav) if x[1] > threshold)
    return wav[index:] + [0] * index

def wav_to_spectral_centroid_bands(fileName, frameTime, paddingSize = 10):
    (sampleRate, audioData, frameSize) = get_wav_data(fileName, frameTime)

    frames = [audioData[i:i+(frameSize)] for i in range(0, min(len(audioData), int(1 / frameTime * paddingSize * frameSize)), (frameSize))]    # group audioData into frames
    if len(frames) < int(sampleRate / frameSize * paddingSize):
        frames += [[0]] * int((sampleRate / frameSize * paddingSize) - len(frames))
    centroids = []
    for frame in frames:
        centroidBands = []
        for f in range(1, len([0, 500, 2500, 5000, 10000, 20000, 100000])):
            centroidBands.append(calculate_spectral_centroid_band(frame, sampleRate, f))
        centroids.append(centroidBands)

    result = []
    for f in range(0, len(centroids[0])):
        for centroid in centroids:
            result.append(centroid[f])

    return result

def calculate_spectral_centroid_band(data, sampleRate, band):
    length = len(data) 
    bands = [0, 500, 2500, 5000, 10000, 20000, 100000]
    fft = list(zip(np.abs(np.fft.fftfreq(length, 1.0/sampleRate)[:length//2+1]), np.abs(np.fft.rfft(data))))
    out = list(zip(*[(f, mag) for (f, mag) in fft if f > bands[band - 1] and f <= bands[band]])) #(freqs, magnitudes)
    if len(out) == 0:
        return 0
    (freqs, magnitudes) = np.array(out[0]), np.array(out[1])
    sums = np.sum(magnitudes)
    return 0 if sums == 0 else np.sum(magnitudes*freqs) / sums                              # return weighted mean

def wav_to_MFCCs(fileName, frameTime, paddingSize = 10):
    audioData, sampleRate = lb.load(fileName)
    #(sampleRate, audioData, frameSize) = get_wav_data(fileName, frameTime)
    jump = 0.5

    beginning = int(sampleRate * jump)
    end = int(paddingSize * sampleRate + sampleRate * jump)

    if len(audioData) < end:
        audioData = audioData + [0] * (end - len(audioData))

    return np.array(lb.feature.mfcc(y=audioData[beginning:end], sr=sampleRate)).flatten()

    frames = [audioData[i:i+(frameSize)] for i in range(0, min(len(audioData), int(1 / frameTime * paddingSize * frameSize)), (frameSize))]    # group audioData into frames
    if len(frames) < int(sampleRate / frameSize * paddingSize):
        frames += [[0]] * int((sampleRate / frameSize * paddingSize) - len(frames))
    
    centroids = [lb.feature.mfcc(frame, sr=sampleRate) for frame in frames]

    return centroids
