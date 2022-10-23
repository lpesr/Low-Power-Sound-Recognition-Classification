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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import pandas as pd
import os
import pickle
import sys
import time

def get_wav_files(dir, label) :
    return [f for f in listdir(dir + "//" + label) if isfile(join(dir + "//" + label, f))]

def prepare_input_data(dir, labels, frameSize) :
    dataVector = []
    labelVector = []
    for label in labels :
        for file in get_wav_files(dir, label) :
            centroids = fe.wav_to_spectral_centroid(dir + "//" + label + "//" + file, frameSize)
            zcr = fe.wav_to_ZCR(dir + "//" + label + "//" + file, frameSize)
            dataVector.append(centroids + zcr)
            labelVector.append(label)
    return (dataVector, labelVector)

(X, Y) = prepare_input_data("..\..\Data", ["glass_breaking", "siren", "crackling_fire", "clapping", "crying_baby"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
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

print("  [Algorithm]----[F-Score]----[Memory Size (KB)]----[Average Elapsed Time (NS)]  ")

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

        st = time.process_time()
        preditctions = clf.predict(X_test)
        et = time.process_time()

        #Caculate the F1 score and store it
        results[j][i] = format(f1_score(y_test, preditctions, average='macro'), ".3f")
        
        i += 1
        totalTime += et - st

    p = pickle.dumps(clf)
    print((names[j], format(sum(map(float, results[j][:])) / len(results[j]), ".3f"), (sys.getsizeof(p) / 1000), ((totalTime * 1000000000) / numKFoldSplits / numKFoldRep / len(X_test))))