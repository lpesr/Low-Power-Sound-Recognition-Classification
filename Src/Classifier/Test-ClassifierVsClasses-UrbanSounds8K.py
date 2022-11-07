from cProfile import label
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
import DataPrep as dp

(X1, Y1) = dp.prepare_input_data_UrbanSounds8K(os.path.join(dirname, "./Data/UrbanSounds8K"), ["drilling"], 0.05, 1.5, 3)
(X2, Y2) = dp.prepare_input_data_UrbanSounds8K(os.path.join(dirname, "./Data/UrbanSounds8K"), ["air_conditioner"], 0.05, 1.5, 3)
(X3, Y3) = dp.prepare_input_data_UrbanSounds8K(os.path.join(dirname, "./Data/UrbanSounds8K"), ["siren"], 0.05, 1.5, 3)
(X4, Y4) = dp.prepare_input_data_UrbanSounds8K(os.path.join(dirname, "./Data/UrbanSounds8K"), ["children_playing"], 0.05, 1.5, 3)
(X5, Y5) = dp.prepare_input_data_UrbanSounds8K(os.path.join(dirname, "./Data/UrbanSounds8K"), ["car_horn"], 0.05, 1.5, 3)
(X6, Y6) = dp.prepare_input_data_UrbanSounds8K(os.path.join(dirname, "./Data/UrbanSounds8K"), ["gun_shot"], 0.05, 1.5, 3)
(X7, Y7) = dp.prepare_input_data_UrbanSounds8K(os.path.join(dirname, "./Data/UrbanSounds8K"), ["engine_idling"], 0.05, 1.5, 3)
(X8, Y8) = dp.prepare_input_data_UrbanSounds8K(os.path.join(dirname, "./Data/UrbanSounds8K"), ["street_music"], 0.05, 1.5, 3)

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

print("  [Algorithm]----[F-Score]----[Memory Size (KB)]----[Average Elapsed Time (uS)]  ")

#Iterate over all of the classifiers
for j in range(len(classifiers)):
    scaler = MinMaxScaler()
    totalTime = 0
    Xbuf = data[0]
    Y = labels[0].copy()

    fScores = ""
    memSize = ""
    elapsedTime = ""

    for k in range(1, len(data)): 
        i = 0
        X = [0] * len(Xbuf)
        Xbuf = [list + data[k][index] for index, list in enumerate(Xbuf)]
        for l in range(0, len(Xbuf)):
            X[l] = scaler.fit_transform(Xbuf[l])
        for l in range(0, len(Y)):
            Y[l] = Y[l] + labels[k][l].copy()

        results = [0 for x in range(len(X1))]

        for i in range(0, 10):
            X_train, X_test = [item for index, sublist in enumerate(X) if index != i for item in sublist], X[i]
            y_train, y_test = [item for index, sublist in enumerate(Y) if index != i for item in sublist], Y[i]

            #Fit the classifier and label the testing split
            clf = classifiers[j].fit(X_train, y_train)

            st = time.time_ns()
            preditctions = clf.predict(X_test)
            et = time.time_ns()

            #Caculate the F1 score and store it
            results[i] = f1_score(y_test, preditctions, average='macro')
            
            i += 1
            totalTime += (et - st) / len(y_test)

        p = pickle.dumps(clf)
        fScores += format(sum(results) / len(results), ".3f") + " & "
        memSize += format(sys.getsizeof(p) / 1000, ".3f") + " & "
        elapsedTime += format((totalTime / 10000), ".3f") + " & "

    print(names[j] + " & " + fScores + memSize + elapsedTime[:-3] + "\\\\")