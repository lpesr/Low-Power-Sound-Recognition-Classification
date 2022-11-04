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

(dataFolds, labelFolds) = prepare_input_data(os.path.join(dirname, "Data/UrbanSounds8K"), ["drilling", "gun_shot", "siren", "children_playing", "car_horn", "air_conditioner", "engine_idling"], 0.068, 3, 0)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"

dirname = os.path.dirname(__file__)

#Define all of the classifiers to test
classifiers = [
    #KNeighborsClassifier(10),
    #SVC(kernel="linear"),
    #SVC(kernel="rbf"),
    #SVC(kernel="sigmoid"),
    #DecisionTreeClassifier(max_depth=100),
    #RandomForestClassifier(max_depth=100, n_estimators=20, max_features=5),
    #MLPClassifier(max_iter=1000),
    GaussianNB(),
]
names = [
    #"Nearest Neighbors",
    #"Linear SVM",
    #"RBF SVM",
    #"Sigmoid SVM",
    #"Decision Tree",
    #"Random Forest",
    #"Neural Net",
    "Naive Bayes",
]

#Create a results matrix
i, j = len(dataFolds), len(classifiers)
results = [[0 for x in range(i)] for y in range(j)] 

#Normalize the data
scaler = MinMaxScaler()
for k in range(0, len(dataFolds)):
    dataFolds[k] = scaler.fit_transform(dataFolds[k])

print("  [Algorithm]----[F-Score]----[Memory Size (KB)]----[Average Elapsed Time (uS)]  ")

#Iterate over all of the classifiers
for j in range(len(classifiers)):
    totalTime = 0

    for i in range(0, 10):
        X_train, X_test = [item for index, sublist in enumerate(dataFolds) if index != i for item in sublist], dataFolds[i]
        y_train, y_test = [item for index, sublist in enumerate(labelFolds) if index != i for item in sublist], labelFolds[i]

        #Fit the classifier and label the testing split
        clf = classifiers[j].fit(X_train, y_train)

        st = time.time_ns()
        preditctions = clf.predict(X_test)
        et = time.time_ns()

        #Caculate the F1 score and store it
        results[j][i] = format(f1_score(y_test, preditctions, average='macro'), ".3f")
        
        totalTime += (et - st) / len(y_test)

    p = pickle.dumps(clf)
    print((names[j], format(sum(map(float, results[j][:])) / len(results[j]), ".3f"), format(sys.getsizeof(p) / 1000, ".3f"), format((totalTime / 10000), ".3f")))