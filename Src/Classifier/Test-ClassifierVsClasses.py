import FeatureExtractor as fe
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

def get_wav_files(dir, label) :
    return [f for f in listdir(dir + "//" + label) if isfile(join(dir + "//" + label, f))]

def prepare_input_data(dir, labels, frameSize) :
    dataVector = []
    labelVector = []
    for label in labels :
        for file in get_wav_files(dir, label) :
            centroids = fe.wav_to_spectral_centroid(dir + "//" + label + "//" + file, frameSize)
            #zcr = fe.wav_to_ZCR(dir + "//" + label + "//" + file, frameSize)
            #dataVector.append(centroids + zcr)
            dataVector.append(centroids)
            labelVector.append(label)
    return (dataVector, labelVector)

(X1, Y1) = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["glass_breaking"], 1000)
(X2, Y2) = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["siren"], 1000)
(X3, Y3) = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["hand_saw"], 1000)
(X4, Y4) = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["crackling_fire"], 1000)
(X5, Y5) = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["clapping"], 1000)
(X6, Y6) = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["crying_baby"], 1000)
(X7, Y7) = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["vacuum_cleaner"], 1000)
(X8, Y8) = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["engine"], 1000)

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
for k in range(0, 8) :
    data[k] = scaler.fit_transform(data[k])

#Create a results matrix
i, j = numKFoldSplits * numKFoldRep, len(classifiers)

#print("  [Algorithm]----[F-Score]----[Memory Size (KB)]----[Average Elapsed Time (MS)]  ")

#Iterate over all of the classifiers
for j in range(len(classifiers)):
    totalTime = 0
    X = data[0]
    Y = labels[0]

    fScores = ""
    memSize = ""
    elapsedTime = ""

    for k in range(1, 8):
        i = 0
        X = np.append(X, data[k], axis=0)
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
            totalTime += et - st

        p = pickle.dumps(clf)
        fScores += format(sum(results) / len(results), ".3f") + " & "
        memSize += format(sys.getsizeof(p) / 1000, ".3f") + " & "
        elapsedTime += format((totalTime / 1000) / (i * len(X_test)), ".3f") + " & "

    print(names[j] + " & " + fScores + memSize + elapsedTime[:-3] + " \\\\")