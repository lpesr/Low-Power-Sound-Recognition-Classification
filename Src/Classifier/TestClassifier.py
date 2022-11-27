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

def plot_features(X, Y):
    buffer = list(zip(X, Y))
    import matplotlib.pyplot as plt

    for point in buffer:
        color = ''
        if point[1] == "glass_breaking":
            color = 'c'
        elif point[1] == "siren":
            color = 'm'
        elif point[1] == "hand_saw":
            color = 'b'
        elif point[1] == "vacuum_cleaner":
            color = 'r'
        elif point[1] == "crackling_fire":
            color = 'g'
        plt.plot(point[0], color)

    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Hz")
    plt.show()

(X, Y) = dp.prepare_input_data(os.path.join(dirname, "Data/Speech-Commands"), ["yes", "no", "on", "off"], 0.05, 1, 4) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
buffer = list(zip(X, Y))
X = np.array(X)
Y = np.array(Y)

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

#Create a results matrix
i, j = numKFoldSplits * numKFoldRep, len(classifiers)
results = [[0 for x in range(i)] for y in range(j)] 

#Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print("  [Algorithm]----[F-Score]----[Memory Size (KB)]----[Average Elapsed Time (uS)]  ")

#Iterate over all of the classifiers
for j in range(len(classifiers)):
    i = 0
    totalTime = 0

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
        results[j][i] = format(f1_score(y_test, preditctions, average='macro'), ".3f")
        
        i += 1
        totalTime += (et - st) / len(y_test)

    p = pickle.dumps(clf)
    print((names[j], format(sum(map(float, results[j][:])) / len(results[j]), ".3f"), format(sys.getsizeof(p) / 1000, ".3f"), format((totalTime / 1000) / i, ".3f")))