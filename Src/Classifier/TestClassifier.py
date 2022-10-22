import FeatureExtractor as fe
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold
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
            dataVector.append(fe.wav_to_spectral_centroid(dir + "//" + label + "//" + file, frameSize))
            labelVector.append(label)
    return (dataVector, labelVector)

(X, Y) = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["airplane", "breathing"], 500)
X = np.array(X)
Y = np.array(Y)

dirname = os.path.dirname(__file__)

#Define all of the classifiers to test
classifiers = [
    SVC(kernel="linear", gamma="auto"),
]
names = [
    "SVM-Linear"
]

numKFoldSplits = 10
numKFoldRep = 2

#Create a results matrix
i, j = numKFoldSplits * numKFoldRep, len(classifiers)
results = [[0 for x in range(i)] for y in range(j)] 

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