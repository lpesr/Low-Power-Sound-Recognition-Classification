import os
from os import listdir
from os.path import isfile, join
import sys
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def get_wav_files(dir, label) :
    return [f for f in listdir(dir + "/" + label) if isfile(join(dir + "/" + label, f))]

def get_avr_amp(dir, labels, numFolds = 5):
    avrAmps = []
    for j, label in enumerate(labels):
        avrAmps.append([0] * 5 * 60)
        for i in range(1, numFolds + 1):
            files = get_wav_files(dir + "/fold" + str(i), label)
            for file in files:
                signal, sampleRate = lb.load(dir + "/fold" + str(i) + "/" + label + "/" + file, sr=25000)
                chunks = np.array_split(signal, len(signal) / sampleRate * 60)
                avr = [(np.mean(list(map(abs, chunk)))) / (len(files) * numFolds) for chunk in chunks]
                avrAmps[j] = [sum(i) for i in zip(avrAmps[j], avr)]  

    return avrAmps

def plot_avr_amp(dir, labels, numFolds = 5):
    avrAmps = get_avr_amp(dir, labels, numFolds)

    for i, label in enumerate(labels):
        plt.plot(avrAmps[i], label=label)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Hz")
    plt.show()

plot_avr_amp(os.path.join(dirname, "Data/ESC-50-Folds"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"])