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

numFolds = 5
#(dataFolds, labelFolds) = dp.prepare_input_data_with_folds(os.path.join(dirname, "./Data/UrbanSounds8k"), ["drilling", "gun_shot", "siren", "children_playing", "car_horn"], 0.05, 0.5, 4)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
(dataFolds, labelFolds) = dp.prepare_input_data_with_folds(os.path.join(dirname, "./Data/ESC-50-Folds"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 00.01, 0.5, 4, numFolds, numFft=512, numMFCC=15, hopLength=512)#dp.prepare_input_data_with_folds(os.path.join(dirname, "./Data/ESC-50-Folds"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 0.01, 0.5, 4, numFolds)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"

dirname = os.path.dirname(__file__)

#Define all of the classifiers to test
classifiers = [
    #KNeighborsClassifier(10),
    #SVC(kernel="linear"),
    #SVC(kernel="rbf"),
    #SVC(kernel="sigmoid"),
    #DecisionTreeClassifier(max_depth=100),
    #RandomForestClassifier(max_depth=100, n_estimators=20, max_features=5),
    GaussianNB(var_smoothing=0.005336699231206307),
    #MLPClassifier(max_iter=1000),
]
names = [
    #"Nearest Neighbors",
    #Linear SVM",
    #"RBF SVM",
    #"Sigmoid SVM",
    #"Decision Tree",
    #"Random Forest",
    "Naive Bayes",
    #"Neural Net",
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

    for i in range(0, numFolds):
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