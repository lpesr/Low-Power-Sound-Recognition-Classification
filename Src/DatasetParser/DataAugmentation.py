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

def compress_dataset(datasetDir, outputDir, labels, targetSamplerate, targetLength):
    if not os.path.exists(outputDir):
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
        apply_data_augmentation(dir, labels, fold="fold" + str(i) + "/")

def apply_data_augmentation(dir, labels, length = 1, fold = ""):
    #folders = ["-0.5-Speed", "-0.75-Speed", "-1.25-Speed", "-1.5-Speed", "--1-pitch", "--0.5-pitch", "-0.5-pitch", "-1-pitch", "-noise", "-inverted", "-randomGain"]
    folders = ["-0.8-Speed", "-0.9-Speed", "-1.1-Speed", "-1.2-Speed", "--4-pitch", "--3-pitch", "--2-pitch", "-2-pitch", "-3-pitch", "-4-pitch"]

    for folder in folders:
        if not os.path.exists(dir + folder):
            os.mkdir(dir + folder)
        #for label in labels:
        #    if not os.path.exists(dir + folder + "/" + label):
        #        os.mkdir(dir + folder + "/" + label)
        #for foldDir in ["/fold1", "/fold2", "/fold3", "/fold4", "/fold5"]:
        #    os.mkdir(dir + folder + foldDir)
        #    for label in labels:
        #        os.mkdir(dir + folder + foldDir + "/" + label)

    for label in labels:
        for file in dp.get_wav_files(dir + "/" + fold[:-1], label):
            audioData, sampleRate = lb.load(dir + "/" + fold + label + "/" + file)
            augmentation_pipeline(audioData, sampleRate, file[:-4], dir, label, fold, length)

def augmentation_pipeline(audioData, sampleRate, filename, dir, label, fold, length = 1):
    sf.write(dir + "-0.8-Speed/" + fold + label + "/" + filename + "0.8-speed.wav", set_audio_length(lb.effects.time_stretch(audioData, rate=0.8), sampleRate, length), sampleRate, format='wav')
    sf.write(dir + "-0.9-Speed/" + fold + label + "/" + filename + "0.9-speed.wav", set_audio_length(lb.effects.time_stretch(audioData, rate=0.9), sampleRate, length), sampleRate, format='wav')
    sf.write(dir + "-1.1-Speed/" + fold + label + "/" + filename + "1.1-speed.wav", set_audio_length(lb.effects.time_stretch(audioData, rate=1.1), sampleRate, length), sampleRate, format='wav')
    sf.write(dir + "-1.2-Speed/" + fold + label + "/" + filename + "1.2-speed.wav", set_audio_length(lb.effects.time_stretch(audioData, rate=1.2), sampleRate, length), sampleRate, format='wav')
    
    sf.write(dir + "--4-pitch/" + fold + label + "/" + filename + "-4-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=-4), sampleRate, length), sampleRate, format='wav')
    sf.write(dir + "--3-pitch/" + fold + label + "/" + filename + "-3-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=-3), sampleRate, length), sampleRate, format='wav')
    sf.write(dir + "--2-pitch/" + fold + label + "/" + filename + "-2-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=-2), sampleRate, length), sampleRate, format='wav')
    sf.write(dir + "-2-pitch/" + fold + label + "/" + filename + "2-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=2), sampleRate, length), sampleRate, format='wav')
    sf.write(dir + "-3-pitch/" + fold + label + "/" + filename + "3-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=3), sampleRate, length), sampleRate, format='wav')
    sf.write(dir + "-4-pitch/" + fold + label + "/" + filename + "4-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=4), sampleRate, length), sampleRate, format='wav')

    #sf.write(dir + "-0.5-Speed/" + fold + label + "/" + filename + "0.5-speed.wav", set_audio_length(lb.effects.time_stretch(audioData, rate=0.5), sampleRate, length), sampleRate, format='wav')
    #sf.write(dir + "-0.75-Speed/" + fold + label + "/" + filename + "0.75-speed.wav", set_audio_length(lb.effects.time_stretch(audioData, rate=0.75), sampleRate, length), sampleRate, format='wav')
    #sf.write(dir + "-1.25-Speed/" + fold + label + "/" + filename + "1.25-speed.wav", set_audio_length(lb.effects.time_stretch(audioData, rate=1.25), sampleRate, length), sampleRate, format='wav')
    #sf.write(dir + "-1.5-Speed/" + fold + label + "/" + filename + "1.5-speed.wav", set_audio_length(lb.effects.time_stretch(audioData, rate=1.5), sampleRate, length), sampleRate, format='wav')
    
    #sf.write(dir + "--1-pitch/" + fold + label + "/" + filename + "-1-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=-1), sampleRate, length), sampleRate, format='wav')
    #sf.write(dir + "--0.5-pitch/" + fold + label + "/" + filename + "-0.5-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=-0.5), sampleRate, length), sampleRate, format='wav')
    #sf.write(dir + "-0.5-pitch/" + fold + label + "/" + filename + "0.5-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=0.5), sampleRate, length), sampleRate, format='wav')
    #sf.write(dir + "-1-pitch/" + fold + label + "/" + filename + "1-pitch.wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=1), sampleRate, length), sampleRate, format='wav')
    
    #sf.write(dir + "-randomGain/" + fold + label + "/" + filename + "gain.wav", set_audio_length(random_gain(audioData), sampleRate, length), sampleRate, format='wav')
    #sf.write(dir + "-noise/" + fold + label + "/" + filename + "noise.wav", set_audio_length(add_white_noise(audioData), sampleRate, length), sampleRate, format='wav')
    #sf.write(dir + "-inverted/" + fold + label + "/" + filename + "inverted.wav", set_audio_length(invert_polarity(audioData), sampleRate, length), sampleRate, format='wav')

def apply_data_augmentation_MFCC(dir, labels, length = 1, nFolds = 5):
    for i in range(1, nFolds + 1):
        fold="fold" + str(i) + "/"

        if not os.path.exists(dir + "-Augmented-MFCC"):
            os.mkdir(dir + "-Augmented-MFCC")
            for foldDir in ["/fold1", "/fold2", "/fold3", "/fold4", "/fold5"]:
                os.mkdir(dir + "-Augmented-MFCC" + foldDir)
                for label in labels:
                    os.mkdir(dir + "-Augmented-MFCC" + foldDir + "/" + label)

        for label in labels:
            for file in dp.get_wav_files(dir + "/" + fold[:-1], label):
                audioData, sampleRate = lb.load(dir + "/" + fold + label + "/" + file)
                augmentation_pipeline_MFCC(audioData, sampleRate, file[:-4], dir, label, fold, length)

def augmentation_pipeline_MFCC(audioData, sampleRate, filename, dir, label, fold, length = 1):
    for i in range(0, 3):
        sf.write(dir + "-Augmented-MFCC/" + fold + label + "/" + filename + "speed-" + str(i) + ".wav", set_audio_length(lb.effects.time_stretch(audioData, rate=np.random.uniform(0.8, 1.1)), sampleRate, length), sampleRate, format='wav')
    
    for i in range(0, 2):
        sf.write(dir + "-Augmented-MFCC/" + fold + label + "/" + filename + "pitch-" + str(i) + ".wav", set_audio_length(lb.effects.pitch_shift(audioData, sr=sampleRate, n_steps=np.random.uniform(-4, -1)), sampleRate, length), sampleRate, format='wav')

    sf.write(dir + "-Augmented-MFCC/" + fold + label + "/" + filename + "gain.wav", set_audio_length(random_gain(audioData), sampleRate, length), sampleRate, format='wav')
    sf.write(dir + "-Augmented-MFCC/" + fold + label + "/" + filename + "inverted.wav", set_audio_length(invert_polarity(audioData), sampleRate, length), sampleRate, format='wav')

#compress_dataset_folds(os.path.join(dirname, "Data/ESC-50-Folds"), os.path.join(dirname, "Data/ESC-50-Folds-Compressed"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 25000, 1, 5)
#apply_data_augmentation_folds(os.path.join(dirname, "Data/ESC-50-Folds-Compressed"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 5)

#compress_dataset(os.path.join(dirname, "Data/Speech-Commands"), os.path.join(dirname, "Data/Speech-Commands-Compressed"), ["yes", "no"], 25000, 1)
#apply_data_augmentation(os.path.join(dirname, "Data/Speech-Commands-Compressed"), ["yes", "no"])

#compress_dataset_urbansounds8k(os.path.join(dirname, "Data/UrbanSounds8k"), os.path.join(dirname, "Data/UrbanSounds8k-Compressed"), ["drilling", "gun_shot", "siren", "children_playing", "car_horn", "air_conditioner", "engine_idling", "street_music"], 25000, 1)
#apply_data_augmentation_urbansounds8k(os.path.join(dirname, "Data/UrbanSounds8k-Compressed"), ["drilling", "gun_shot", "siren", "children_playing", "car_horn", "air_conditioner", "engine_idling", "street_music"])

apply_data_augmentation_MFCC(os.path.join(dirname, "Data/ESC-50-Folds-Compressed"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 1, 5)