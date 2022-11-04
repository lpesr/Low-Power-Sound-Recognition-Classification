from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import pandas as pd
import os
import pickle
import sys
import time

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.join(dirname, 'Src/FeatureExtractor'))
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
    """
    dataFolds = []
    labelFolds = []
    for i in range(1, 11):
        dataVector = []
        labelVector = []
        for label in labels:
            for file in get_wav_files(dir + "/fold" + str(i), label):
                try:
                    if featureType == 0:
                        centroids = fe.wav_to_spectral_centroid(dir  + "/fold" + str(i) + "/" + label + "/" + file, frameTime, wavLength)
                        dataVector.append(centroids)
                    elif featureType == 1:
                        zcr = fe.wav_to_ZCR(dir + "/fold" + str(i) + "/" + label + "/" + file, frameTime, wavLength)
                        dataVector.append(zcr)
                    elif featureType == 2:
                        centroids = fe.wav_to_spectral_centroid(dir  + "/fold" + str(i) + "/" + label + "/" + file, frameTime, wavLength)
                        zcr = fe.wav_to_ZCR(dir + "/fold" + str(i) + "/" + label + "/" + file, frameTime, wavLength)
                        dataVector.append(centroids + zcr)
                    else:
                        raise Exception("Error: Not valid feature ID")
                    labelVector.append(label)
                except Exception as error:
                    print('Caught this error: ' + repr(error))
        dataFolds.append(dataVector)
        labelFolds.append(labelVector)
    return (dataFolds, labelFolds)


def train_classifier(labels, classifier, frameTime, folds, wavLength, featureType):
    """Train and return a given classifier
    Prams:
        Labels = Array of strings for the diffrent disired labels to train on
        Classifier = A given classifier to train on
        Frame Time = 1/the number of frames per second
        Folds = An array of the fold numbers to train on
        wavLength = length in seconds for the wav file to be shortened/snipped (or extended) to
        featureType = 0 - centroids
                      1 - Zero Crossing Rate
                      2 - centroids + Zero Crossing Rate
    Example:
        train_classifier(["drilling", "gun_shot", "siren"], GaussianNB(), 0.05, [0,2,3,4,5,6,7,8], 3, 0)
        This will train a Naive Bayes classifier on the classes drilling, gun shot, and siren
        using all of the folds but the 10th fold. The WAV files would also be cut to 3 seconds long and
        trained on spectral centroids
    """
    (dataFolds, labelFolds) = prepare_input_data(os.path.join(dirname, "Data/UrbanSounds8K"), labels, frameTime, wavLength, featureType)

    #Normalize the data
    scaler = MinMaxScaler()
    for k in range(0, len(dataFolds)):
        dataFolds[k] = scaler.fit_transform(dataFolds[k])

    X_train, X_test = [item for index in folds for item in dataFolds[index]], [item for index in [i for i in list(range(0, 10)) if i not in folds] for item in dataFolds[index]]
    y_train, y_test = [item for index in folds for item in labelFolds[index]], [item for index in [i for i in list(range(0, 10)) if i not in folds] for item in labelFolds[index]]

    st = time.time_ns()
    clf = classifier.fit(X_train, y_train)
    et = time.time_ns()

    preditctions = clf.predict(X_test)
    print("F-Score: " + format(f1_score(y_test, preditctions, average='macro'), ".3f"))
    print("Size (KB): " + format(sys.getsizeof(pickle.dumps(clf)) / 1000, ".3f"))
    print("Predition Time (uS): " + format(((et - st) / 1000) / len(y_test), ".3f"))

    return clf

#train_classifier(["drilling", "gun_shot", "siren"], GaussianNB(), 0.05, [1,2,3,4,5,6,7,8], 3, 0)