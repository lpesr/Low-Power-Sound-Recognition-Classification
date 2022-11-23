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
    audioData = lb.resample(audioData, orig_sr=sampleRate, target_sr=targetSamplerate)
    return set_audio_length(audioData, sampleRate, targetLength)

def compress_dataset(datasetDir, outputDir, labels, targetSamplerate, targetLength):
    os.mkdir(outputDir)
    for label in labels:
        os.mkdir(outputDir + "/" + label)
        for file in dp.get_wav_files(datasetDir, label):
            compressedWav = compress_wav_file(datasetDir + "/" + label + "/" + file, targetSamplerate, targetLength)
            sf.write(outputDir + "/" + label + "/" + file, compressedWav, targetSamplerate, format='wav')

def add_white_noise(data, noise_percentage_factor = 0.1):
    noise = np.random.normal(0, data.std(), data.size)
    return data + noise * noise_percentage_factor

def random_gain(data, min_factor = 0.1, max_factor = 0.12):
    gain_rate = np.random.uniform(min_factor, max_factor)
    return data * gain_rate

def invert_polarity(data):
    return data * -1

def apply_data_augmentation(dir, labels):
    for label in labels:
        for file in dp.get_wav_files(dir, label):
            outputBaseFilePath = dir + "/" + label + "/" + file[:-4]
            audioData, sampleRate = lb.load(dir + "/" + label + "/" + file)
            sf.write(outputBaseFilePath + "noise.wav", add_white_noise(audioData), sampleRate, format='wav')
            sf.write(outputBaseFilePath + "random_gain.wav", random_gain(audioData), sampleRate, format='wav')
            sf.write(outputBaseFilePath + "invert_polarity.wav", invert_polarity(audioData), sampleRate, format='wav')
            sf.write(outputBaseFilePath + "invert_polarity.wav", invert_polarity(audioData), sampleRate, format='wav')
            sf.write(outputBaseFilePath + "slow.wav", lb.effects.time_stretch(audioData, rate=0.5), sampleRate, format='wav')
            sf.write(outputBaseFilePath + "fast.wav", lb.effects.time_stretch(audioData, rate=1.5), sampleRate, format='wav')
            sf.write(outputBaseFilePath + "high.wav", lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=1), sampleRate, format='wav')
            sf.write(outputBaseFilePath + "low.wav", lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=-1), sampleRate, format='wav')

compress_dataset(os.path.join(dirname, "Data/ESC-50"), os.path.join(dirname, "Data/ESC-50-Compressed"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 25000, 1)
apply_data_augmentation(os.path.join(dirname, "Data/ESC-50-Compressed"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"])