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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score
import pandas as pd
import os
import pickle
import sys
import time
import matplotlib.pyplot as plt

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
                else:
                    raise Exception("Error: Not valid feature ID")
                labelVector.append(label)
            except Exception as error:
                print('Caught this error: ' + repr(error))
    return (dataVector, labelVector)

(X1, Y1) = prepare_input_data(os.path.join(dirname, "Data\ESC-50"), ["glass_breaking"], 0.02, 3, 0)
(X2, Y2) = prepare_input_data(os.path.join(dirname, "Data\ESC-50"), ["siren"], 0.02, 3, 0)
(X3, Y3) = prepare_input_data(os.path.join(dirname, "Data\ESC-50"), ["hand_saw"], 0.02, 3, 0)
(X4, Y4) = prepare_input_data(os.path.join(dirname, "Data\ESC-50"), ["crackling_fire"], 0.02, 3, 0)
(X5, Y5) = prepare_input_data(os.path.join(dirname, "Data\ESC-50"), ["clapping"], 0.02, 3, 0)
(X6, Y6) = prepare_input_data(os.path.join(dirname, "Data\ESC-50"), ["crying_baby"], 0.02, 3, 0)
(X7, Y7) = prepare_input_data(os.path.join(dirname, "Data\ESC-50"), ["vacuum_cleaner"], 0.02, 3, 0)
(X8, Y8) = prepare_input_data(os.path.join(dirname, "Data\ESC-50"), ["engine"], 0.02, 3, 0)

data = [X1, X2, X3, X4, X5, X6, X7, X8]
labels = [Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8]

dirname = os.path.dirname(__file__)

#Define all of the classifiers to test
classifiers = [
    KNeighborsClassifier(10),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    SVC(kernel="sigmoid"),
    DecisionTreeClassifier(max_depth=100),
    RandomForestClassifier(max_depth=100, n_estimators=20, max_features=5),
    MLPClassifier(max_iter=1000),
    GaussianNB(),
]
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Sigmoid SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "Naive Bayes",
]

numKFoldSplits = 10
numKFoldRep = 2

#Normalize the data
scaler = MaxAbsScaler()

#Create a results matrix
i, j = numKFoldSplits * numKFoldRep, len(classifiers)

#print("  [Algorithm]----[F-Score]----[Memory Size (KB)]----[Average Elapsed Time (uS)]  ")

#Iterate over all of the classifiers
for j in range(len(classifiers)):
    totalTime = 0
    Xbuf = data[0]
    Y = labels[0]

    fScores = ""
    memSize = ""
    elapsedTime = ""

    for k in range(1, 8): 
        i = 0
        Xbuf = np.append(Xbuf, data[k], axis=0)
        X = scaler.fit_transform(Xbuf)
        Y = np.append(Y, labels[k], axis=0)

        results = [0 for x in range(numKFoldSplits * numKFoldRep)]

        #K cross validate the data (there will be an equal number of both classes to train on)
        #This is because the data was split and then combined earlier
        kf = RepeatedKFold(n_splits=numKFoldSplits, n_repeats=numKFoldRep)
        for train, test in kf.split(X):
            X_train, X_test = X[train], X[test]
            y_train, y_test = Y[train], Y[test]

            #Fit the classifier and label the testing split
            clf = classifiers[j].fit(X_train, y_train)

            st = time.time_ns()
            preditctions = clf.predict(X_test)
            et = time.time_ns()

            #Caculate the F1 score and store it
            results[i] = f1_score(y_test, preditctions, average='macro')
            
            i += 1
            totalTime += (et - st) / len(y_train)

        p = pickle.dumps(clf)
        fScores += format(sum(results) / len(results), ".3f") + " & "
        memSize += format(sys.getsizeof(p) / 1000, ".3f") + " & "
        elapsedTime += format((totalTime / 1000) / i, ".3f") + " & "

    print(names[j] + " & " + fScores + memSize + elapsedTime[:-3] + "\\\\")