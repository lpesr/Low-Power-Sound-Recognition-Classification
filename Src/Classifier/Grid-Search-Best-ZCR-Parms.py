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
from itertools import combinations

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.join(dirname, 'Src/FeatureExtractor'))
import DataPrep as dp

#Define all of the classifiers to test
classifier = GaussianNB()

def run_grid_search(originalData, augDataset, frameLengths, fileLengths, outputFile):
    output = open(outputFile, "a")

    output.write("FileLength")
    for frameLength in frameLengths:
        output.write("," + str(frameLength))
    output.write("\n")

    for fileLength in fileLengths:
        output.write(str(fileLength) + ":")
        for frameLength in frameLengths:
            output.write(",")
            result = test_dataset(originalData, augDataset, frameLength, fileLength)
            output.write(str(result))
        output.write("\n")
        output.flush()
    output.close()
            
def test_dataset(originalDataset, augmentedDataset, fameLength, fileLength):
    numFolds = 5
    (dataFolds, labelFolds) = dp.prepare_input_data_with_folds(originalDataset, ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], fameLength, fileLength, 1, numFolds)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
    #(orignialDataFolds, orignialLabelFolds) = (dataFolds, labelFolds)

    #(dataFoldsBuffer, labelFoldsBuffer) = dp.prepare_input_data_with_folds(originalDataset + augmentedDataset, ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], fameLength, fileLength, 1, numFolds)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
    #for fold in range(0, numFolds):
    #    dataFolds[fold] += dataFoldsBuffer[fold]
    #    labelFolds[fold] += labelFoldsBuffer[fold]

    #Create a results matrix
    i = len(dataFolds)
    results = [0 for x in range(i)]

    #Normalize the data
    scaler = MinMaxScaler()
    for k in range(0, len(dataFolds)):
        dataFolds[k] = scaler.fit_transform(dataFolds[k])

    for i in range(0, numFolds):
        X_train, X_test = [item for index, sublist in enumerate(dataFolds) if index != i for item in sublist], dataFolds[i]
        y_train, y_test = [item for index, sublist in enumerate(labelFolds) if index != i for item in sublist], labelFolds[i]

        #Fit the classifier and label the testing split
        clf = classifier.fit(X_train, y_train)

        preditctions = clf.predict(X_test)

        #Caculate the F1 score and store it
        results[i] = format(f1_score(y_test, preditctions, average='macro'), ".3f")

    p = pickle.dumps(clf)
    return (format(sum(map(float, results)) / len(results), ".3f"), format(sys.getsizeof(p) / 1000, ".3f"))

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

originalDataset = os.path.join(dirname, "Data/ESC-50-Folds")
run_grid_search(originalDataset, "-Augmented-MFCC", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4], "U:/GDP/ML Testing/Low-Power-Sound-Recognition-Classification/Src/ResultsFormatter/grid_search-ESC-50-ZCR.csv")