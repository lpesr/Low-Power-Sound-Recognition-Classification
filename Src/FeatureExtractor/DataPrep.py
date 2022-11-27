import os
from os import listdir
from os.path import isfile, join
import sys
import FeatureExtractor as fe

def get_wav_files(dir, label) :
    return [f for f in listdir(dir + "/" + label) if isfile(join(dir + "/" + label, f))]

def prepare_input_data(dir, labels, frameTime, wavLength, featureType):
    """Extract the feature vectors from the WAV files
    Prams:
        dir = dataset dir
        labels = labels to train on
        Frame Time = 1/the number of frames per second
        wavLength = length in seconds for the wav file to be shortened/snipped (or extended) to
        featureType = 0 - centroids
                      1 - Zero Crossing Rate
                      2 - centroids + Zero Crossing Rate
                      3 - centroid bands
                      4 - MFCCs
    """
    dataVector = []
    labelVector = []
    for label in labels:
        for file in get_wav_files(dir, label):
            try:
                if featureType == 0:
                    centroids = fe.wav_to_spectral_centroid(dir  + "/" + label + "/" + file, frameTime, wavLength)
                    dataVector.append(centroids)
                elif featureType == 1:
                    zcr = fe.wav_to_ZCR(dir  + "/" + label + "/" + file, frameTime, wavLength)
                    dataVector.append(zcr)
                elif featureType == 2:
                    centroids = fe.wav_to_spectral_centroid(dir  + "/" + label + "/" + file, frameTime, wavLength)
                    zcr = fe.wav_to_ZCR(dir  + "/" + label + "/" + file, frameTime, wavLength)
                    dataVector.append(centroids + zcr)
                elif featureType == 3:
                    centroids = fe.wav_to_spectral_centroid_bands(dir  + "/" + label + "/" + file, frameTime, wavLength)
                    dataVector.append(centroids)
                elif featureType == 4:
                    mfccs = fe.wav_to_MFCCs(dir  + "/" + label + "/" + file, frameTime, wavLength, jump=0)
                    #for mfcc in mfccs:
                    #    dataVector.append(mfcc)
                    dataVector.append(mfccs)
                else:
                    raise Exception("Error: Not valid feature ID")
                labelVector.append(label)
            except Exception as error:
                print('Caught this error: ' + repr(error))
    return (dataVector, labelVector)

def prepare_input_data_UrbanSounds8K(dir, labels, frameTime, wavLength, featureType, numFolds=10):
    dataFolds = []
    labelFolds = []
    for i in range(1, numFolds + 1):
        (dataVector, labelVector) = prepare_input_data(dir + "/fold" + str(i), labels, frameTime, wavLength, featureType)
        dataFolds.append(dataVector)
        labelFolds.append(labelVector)
    return (dataFolds, labelFolds)