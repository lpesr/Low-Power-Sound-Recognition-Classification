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
import librosa as lb

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.join(dirname, 'Src/FeatureExtractor'))
sys.path.append(os.path.join(dirname, 'Src/DatasetParser'))
import DataPrep as dp
import DataAugmentation as da

def train_classifier(testFold, labels = ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"],
                     originalDataset = os.path.join(dirname, "Data/ESC-50-Folds-Compressed"), 
                     augmentedDataset = os.path.join(dirname, "Data/ESC-50-Folds-Augmented-MFCC"), numFfts = 512, numMfcc = 15, 
                     wavLength = 0.5, numFolds = 5, classifier = GaussianNB(var_smoothing=0.005336699231206307)):
    """Train and return a given classifier
    Prams:
        testFold = The fold that won't be trained on and should be tested on
        labels = Array of strings for the diffrent disired labels to train on
        originalDataset = Compressed (cut to size and resampled) dataset but not augmented dir
        augmentedDataset = Augmented dataset dir
        numFfts = number of ffts in a frame (frame size)
        numMfcc = number of MFCC coeficientes
        classifier = A given classifier to train on
        numFolds = total number of folds (Don't change this if using ESC-50 because it has 5 folds)
        wavLength = length in seconds for the wav file
    Example:
        train_classifier(4)
        This will train a Naive Bayes classifier on the classes ["glass_breaking", "siren", "hand_saw", 
        "vacuum_cleaner", "crackling_fire"] using all of the folds but the 5th fold.
    """
    
    (dataFolds, labelFolds) = dp.prepare_input_data_with_folds(originalDataset, labels, 0, wavLength, 4, numFolds, numFft=numFfts, numMFCC=numMfcc, hopLength=numFfts)
    (orignialDataFolds, orignialLabelFolds) = (dataFolds, labelFolds)

    (dataFoldsBuffer, labelFoldsBuffer) = dp.prepare_input_data_with_folds(augmentedDataset, labels, 0, wavLength, 4, numFolds, numFft=numFfts, numMFCC=numMfcc, hopLength=numFfts)
    for fold in range(0, numFolds):
        dataFolds[fold] += dataFoldsBuffer[fold]
        labelFolds[fold] += labelFoldsBuffer[fold]

    #Normalize the data
    scaler = MinMaxScaler()
    for k in range(0, len(dataFolds)):
        dataFolds[k] = scaler.fit_transform(dataFolds[k])

    X_train, X_test = [item for fold in range(0, numFolds) if fold != testFold for item in dataFolds[fold]], orignialDataFolds[testFold]
    y_train, y_test = [item for fold in range(0, numFolds) if fold != testFold for item in labelFolds[fold]], orignialLabelFolds[testFold]

    clf = classifier.fit(X_train, y_train)
    
    st = time.time_ns()
    preditctions = clf.predict(X_test)
    et = time.time_ns()

    print("F-Score: " + format(f1_score(y_test, preditctions, average='macro'), ".3f"))
    print("Size (KB): " + format(sys.getsizeof(pickle.dumps(clf)) / 1000, ".3f"))
    print("Predition Time (uS): " + format(((et - st) / 1000) / len(y_test), ".3f"))

    return clf

"""
Only use the following functions if you haven't yet created the compressed dataset and the augmented dataset
"""
#da.compress_dataset_folds(os.path.join(dirname, "Data/ESC-50-Folds"), os.path.join(dirname, "Data/ESC-50-Folds-Compressed"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], targetSamplerate=25000, targetLength=0.5, nFolds=5)
#da.apply_data_augmentation_MFCC(os.path.join(dirname, "Data/ESC-50-Folds"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], length=0.5)

for fold in range(0, 5):
    train_classifier(fold)