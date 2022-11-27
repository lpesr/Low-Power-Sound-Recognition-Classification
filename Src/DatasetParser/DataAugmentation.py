import os
import sys
import numpy as np
import librosa as lb
import soundfile as sf

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.join(dirname, 'Src/FeatureExtractor'))
import DataPrep as dp
import FeatureExtractor as fe

def set_audio_length(singleBandWav, samplerate, length):
    numOfSamples = int(samplerate * length)
    if len(singleBandWav) > numOfSamples:
        return singleBandWav[:numOfSamples]
    elif len(singleBandWav) < numOfSamples:
        return np.append(singleBandWav, [0.0] * int(numOfSamples - len(singleBandWav)))
    else:
        return singleBandWav

def compress_wav_file(filename, targetSamplerate, targetLength):
    audioData, sampleRate = lb.load(filename)
    audioData = fe.convert_to_single_band(audioData)
    audioData = lb.resample(audioData, orig_sr=sampleRate, target_sr=targetSamplerate)
    return set_audio_length(audioData, sampleRate, targetLength)

def compress_dataset_folds(datasetDir, outputDir, labels, targetSamplerate, targetLength, nFolds=10):
    os.mkdir(outputDir)
    for i in range(1, nFolds + 1):
        os.mkdir(outputDir + "/fold" + str(i))
        for label in labels:
            os.mkdir(outputDir + "/fold" + str(i) + "/" + label)
            for file in dp.get_wav_files(datasetDir + "/fold" + str(i), label):
                compressedWav = compress_wav_file(datasetDir + "/fold" + str(i) + "/" + label + "/" + file, targetSamplerate, targetLength)
                sf.write(outputDir + "/fold" + str(i) + "/" + label + "/" + file, compressedWav, targetSamplerate, format='wav')

def compress_dataset_esc(datasetDir, outputDir, labels, targetSamplerate, targetLength):
    os.mkdir(outputDir)
    for label in labels:
        os.mkdir(outputDir + "/" + label)
        for file in dp.get_wav_files(datasetDir, label):
            compressedWav = compress_wav_file(datasetDir + "/" + label + "/" + file, targetSamplerate, targetLength)
            sf.write(outputDir + "/" + label + "/" + file, compressedWav, targetSamplerate, format='wav')

def add_white_noise(data, noise_percentage_factor = 0.1):
    noise = np.random.normal(0, data.std(), data.size)
    return data + noise * noise_percentage_factor

def random_gain(data, min_factor = 0.5, max_factor = 1):
    gain_rate = np.random.uniform(min_factor, max_factor)
    return data * gain_rate

def invert_polarity(data):
    return data * -1

def apply_data_augmentation_folds(dir, labels, nFolds):
    for i in range(1, nFolds + 1):
        apply_data_augmentation(dir + "/fold" + str(i), labels)

def apply_data_augmentation(dir, labels, length = 1):
    for label in labels:
        for file in dp.get_wav_files(dir, label):
            outputBaseFilePath = dir + "/" + label + "/" + file[:-4]
            audioData, sampleRate = lb.load(dir + "/" + label + "/" + file)
            augWavFiles = augmentation_pipeline(audioData, sampleRate)
            for file in augWavFiles:
                sf.write(outputBaseFilePath + file[1], set_audio_length(file[0], sampleRate, length), sampleRate, format='wav')

def augmentation_pipeline(audioData, sampleRate):
    wavTimeStretchFiles = []
    wavTimeStretchFiles.append((lb.effects.time_stretch(audioData, rate=0.5), "-0.5speed"))
    #wavTimeStretchFiles.append((lb.effects.time_stretch(audioData, rate=0.75), "-0.75speed"))
    wavTimeStretchFiles.append((audioData, "-1speed"))
    #wavTimeStretchFiles.append((lb.effects.time_stretch(audioData, rate=1.25), "-1.25speed"))
    wavTimeStretchFiles.append((lb.effects.time_stretch(audioData, rate=1.5), "-1.5speed"))

    wavPitchFiles = []
    for file in wavTimeStretchFiles:
        wavPitchFiles.append((lb.effects.pitch_shift(file[0], sr=sampleRate, n_steps=-1), file[1] + "-dsift1"))
        #wavPitchFiles.append((lb.effects.pitch_shift(file[0], sr=sampleRate, n_steps=-0.5), file[1] + "-dsift0.5"))
        wavPitchFiles.append((file[0], file[1] + "-nosift"))
        #wavPitchFiles.append((lb.effects.pitch_shift(file[0], sr=sampleRate, n_steps=0.5), file[1] + "-usift0.5"))
        wavPitchFiles.append((lb.effects.pitch_shift(file[0], sr=sampleRate, n_steps=1), file[1] + "-usift1"))
    
    wavFiles = []
    for file in wavPitchFiles:
        wavFiles.append((random_gain(add_white_noise(file[0])), file[1] + ".wav"))
        wavFiles.append((random_gain(add_white_noise(invert_polarity(file[0]))), file[1] + "-inverted.wav"))

    return wavFiles

#compress_dataset_folds(os.path.join(dirname, "Data/ESC-50-Folds"), os.path.join(dirname, "Data/ESC-50-Folds10-Compressed"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 25000, 1, 5)
apply_data_augmentation_folds(os.path.join(dirname, "Data/ESC-50-Folds-Compressed"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 10)

#compress_dataset_urbansounds8k(os.path.join(dirname, "Data/UrbanSounds8k"), os.path.join(dirname, "Data/UrbanSounds8k-Compressed"), ["drilling", "gun_shot", "siren", "children_playing", "car_horn", "air_conditioner", "engine_idling", "street_music"], 25000, 1)
#apply_data_augmentation_urbansounds8k(os.path.join(dirname, "Data/UrbanSounds8k-Compressed"), ["drilling", "gun_shot", "siren", "children_playing", "car_horn", "air_conditioner", "engine_idling", "street_music"])