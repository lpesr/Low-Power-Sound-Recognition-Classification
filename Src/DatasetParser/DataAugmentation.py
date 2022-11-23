import os
import sys
import numpy as np
import librosa as lb
import soundfile as sf

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.join(dirname, 'Src/FeatureExtractor'))
import DataPrep as dp

def set_audio_length(singleBandWav, samplerate, length):
    numOfSamples = int(samplerate * length)
    if len(singleBandWav) > numOfSamples:
        return singleBandWav[:numOfSamples]
    elif len(singleBandWav) < numOfSamples:
        return singleBandWav + [0] * int(numOfSamples - len(singleBandWav))
    else:
        return singleBandWav

def compress_wav_file(filename, targetSamplerate, targetLength):
    audioData, sampleRate = lb.load(filename)
    audioData = set_audio_length(audioData, sampleRate, targetLength)
    return lb.resample(audioData, orig_sr=sampleRate, target_sr=targetSamplerate)

def compress_dataset(datasetDir, outputDir, labels, targetSamplerate, targetLength):
    os.mkdir(outputDir)
    for label in labels:
        os.mkdir(outputDir + "/" + label)
        for file in dp.get_wav_files(datasetDir, label):
            compressedWav = compress_wav_file(datasetDir + "/" + label + "/" + file, targetSamplerate, targetLength)
            sf.write(outputDir + "/" + label + "/" + file, compressedWav, targetSamplerate, format='wav')

def add_white_noise(data, noise_percentage_factor):
    noise = np.random.normal(0, data.std(), data.size)
    return data + noise * noise_percentage_factor

compress_dataset(os.path.join(dirname, "Data/ESC-50"), os.path.join(dirname, "Data/ESC-50-Compressed"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 25000, 1)