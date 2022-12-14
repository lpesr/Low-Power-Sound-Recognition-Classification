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
import warnings

warnings.filterwarnings("ignore")

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.join(dirname, 'Src/FeatureExtractor'))
import DataPrep as dp

#["glass_breaking", "siren", "hand_saw", "crackling_fire", "clapping", "crying_baby", "vacuum_cleaner", "engine"]

def test_feature(dataset, labels, feature):
    dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    (X1, Y1) = dp.prepare_input_data(os.path.join(dirname, dataset), [labels[0]], 0.02, 1, feature)
    (X2, Y2) = dp.prepare_input_data(os.path.join(dirname, dataset), [labels[1]], 0.02, 1, feature)
    (X3, Y3) = dp.prepare_input_data(os.path.join(dirname, dataset), [labels[2]], 0.02, 1, feature)
    (X4, Y4) = dp.prepare_input_data(os.path.join(dirname, dataset), [labels[3]], 0.02, 1, feature)
    (X5, Y5) = dp.prepare_input_data(os.path.join(dirname, dataset), [labels[4]], 0.02, 1, feature)
    (X6, Y6) = dp.prepare_input_data(os.path.join(dirname, dataset), [labels[5]], 0.02, 1, feature)
    (X7, Y7) = dp.prepare_input_data(os.path.join(dirname, dataset), [labels[6]], 0.02, 1, feature)
    (X8, Y8) = dp.prepare_input_data(os.path.join(dirname, dataset), [labels[7]], 0.02, 1, feature)

    data = [X1, X2, X3, X4, X5, X6, X7, X8]
    labels = [Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8]

    dirname = os.path.dirname(__file__)

    #Define all of the classifiers to test
    classifiers = [
        #KNeighborsClassifier(),
        #SVC(kernel="linear", cache_size=7000),
        #SVC(kernel="rbf", cache_size=7000),
        #SVC(kernel="sigmoid", cache_size=7000),
        DecisionTreeClassifier(max_depth=100),
        RandomForestClassifier(max_depth=100, n_estimators=20, max_features=5),
        MLPClassifier(max_iter=1000),
        GaussianNB(),
    ]
    names = [
        #"Nearest Neighbors",
        #"Linear SVM",
        #"RBF SVM",
        #"Sigmoid SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "Naive Bayes",
    ]

    numKFoldSplits = 10
    numKFoldRep = 2

    #Normalize the data
    scaler = MaxAbsScaler()

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

def run_test_rig(dataset, labels):
    features = ["centroids", "Zero Crossing Rate", "centroids + Zero Crossing Rate", "centroid bands", "MFCCs"]
    for i in range(0, len(features)):
        print(features[i])
        test_feature(dataset, labels, i)

run_test_rig("Data/Speech-Commands", ["yes", "no", "on", "off", "go", "stop", "up", "down"])