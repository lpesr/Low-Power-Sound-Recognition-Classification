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
classifiers = [
    SVC(kernel="rbf"),
    GaussianNB(),
    MLPClassifier(max_iter=1000),
]
names = [
    "RBF SVM",
    "Naive Bayes",
    "Neural Net",
]

def run_evaluation(originalData, augDatasets, outputFile):
    output = open(outputFile, "a")

    output.write("Augmented Data," + names[0] + "," + names[1] + "," + names[2] + "\n")

    for datasets in augDatasets:
        for L in range(len(datasets) + 1):
            for comb in combinations(datasets, L):
                test_augmented_dataset(originalData, comb, output)
    output.close()
            
def test_augmented_dataset(originalDataset, combinations, outputFile):
    if len(combinations) == 0:
        return

    numFolds = 5
    (dataFolds, labelFolds) = dp.prepare_input_data_with_folds(originalDataset, ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 0.05, 0.5, 4, numFolds)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
    (orignialDataFolds, orignialLabelFolds) = (dataFolds, labelFolds)

    for combination in combinations:
        (dataFoldsBuffer, labelFoldsBuffer) = dp.prepare_input_data_with_folds(originalDataset + combination, ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 0.05, 0.5, 4, numFolds)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
        for fold in range(0, numFolds):
            dataFolds[fold] += dataFoldsBuffer[fold]
            labelFolds[fold] += labelFoldsBuffer[fold]

    #Create a results matrix
    i, j = len(dataFolds), len(classifiers)
    results = [[0 for x in range(i)] for y in range(j)] 

    #Normalize the data
    scaler = MinMaxScaler()
    for k in range(0, len(dataFolds)):
        dataFolds[k] = scaler.fit_transform(dataFolds[k])

    #Iterate over all of the classifiers
    for j in range(len(classifiers)):
        for i in range(0, numFolds):
            X_train, X_test = [item for index, sublist in enumerate(dataFolds) if index != i for item in sublist], orignialDataFolds[i]
            y_train, y_test = [item for index, sublist in enumerate(labelFolds) if index != i for item in sublist], orignialLabelFolds[i]

            #Fit the classifier and label the testing split
            clf = classifiers[j].fit(X_train, y_train)

            preditctions = clf.predict(X_test)

            #Caculate the F1 score and store it
            results[j][i] = format(f1_score(y_test, preditctions, average='macro'), ".3f")

    outputFile.write("Original + " + " ".join([name[1:] for name in combinations]) + "," + format(sum(map(float, results[0][:])) / len(results[j]), ".3f") + "," + format(sum(map(float, results[1][:])) / len(results[j]), ".3f") + "," + format(sum(map(float, results[2][:])) / len(results[j]), ".3f") + "\n")
    outputFile.flush()

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

originalDataset = os.path.join(dirname, "Data/ESC-50-Folds-Compressed")
augmnetedSpeedDatasets = ["-0.8-Speed", "-0.9-Speed", "-1.1-Speed", "-1.2-Speed"]
augmentedPitchDatasets = ["--4-pitch", "--3-pitch", "--2-pitch", "--1-pitch", "-1-pitch", "-2-pitch", "-3-pitch", "-4-pitch"]
#augmentedNoiseDataset = ["-noise"]
#augmentedInvertedDataset = ["-inverted"]
#augmentedGainDataset = ["-randomGain"]
#augmentedDatasets = [["-0.5-pitch", "-1-pitch", "-randomGain", "-inverted", "-0.5-Speed", "-0.75-Speed"]]
augmentedDatasets = [augmnetedSpeedDatasets, augmentedPitchDatasets]#, augmentedNoiseDataset, augmentedInvertedDataset, augmentedGainDataset]

run_evaluation(originalDataset, augmentedDatasets, "U:/GDP/ML Testing/Low-Power-Sound-Recognition-Classification/Src/ResultsFormatter/augmentationOutputESC-50-MFCCs.csv")