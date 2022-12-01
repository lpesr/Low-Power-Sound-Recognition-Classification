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

def run_grid_search(originalData, augDataset, numFFTs, numMFCCs, fileLengths, outputFile):
    output = open(outputFile, "a")

    for numMFCC in numMFCCs:
        output.write("numMFCCs: " + str(numMFCC) + ",\n")

        output.write("FileLength")
        for fileLength in fileLengths:
            output.write("," + str(fileLength))
        output.write("\n")

        for numFFT in numFFTs:
            output.write(str(numFFT) + ":")
            for fileLength in fileLengths:
                output.write(",")
                result = test_dataset(originalData, augDataset, numFFT, numMFCC, fileLength)
                output.write(str(result[0]) + " : " + str(result[1]))
            output.write("\n")
            output.flush()

        output.write("\n\n")
    output.close()
            
def test_dataset(originalDataset, augmentedDataset, numFFT, numMFCC, fileLength):
    numFolds = 5
    (dataFolds, labelFolds) = dp.prepare_input_data_with_folds(originalDataset, ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 0, fileLength, 4, numFolds, numFft=numFFT, numMFCC=numMFCC)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
    (orignialDataFolds, orignialLabelFolds) = (dataFolds, labelFolds)

    (dataFoldsBuffer, labelFoldsBuffer) = dp.prepare_input_data_with_folds(originalDataset + augmentedDataset, ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 0, fileLength, 4, numFolds, numFft=numFFT, numMFCC=numMFCC)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
    for fold in range(0, numFolds):
        dataFolds[fold] += dataFoldsBuffer[fold]
        labelFolds[fold] += labelFoldsBuffer[fold]

    #Create a results matrix
    i = len(dataFolds)
    results = [0 for x in range(i)]

    #Normalize the data
    scaler = MinMaxScaler()
    for k in range(0, len(dataFolds)):
        dataFolds[k] = scaler.fit_transform(dataFolds[k])
    for k in range(0, len(orignialDataFolds)):
        orignialDataFolds[k] = scaler.fit_transform(orignialDataFolds[k])

    for i in range(0, numFolds):
        X_train, X_test = [item for index, sublist in enumerate(dataFolds) if index != i for item in sublist], orignialDataFolds[i]
        y_train, y_test = [item for index, sublist in enumerate(labelFolds) if index != i for item in sublist], orignialLabelFolds[i]

        #Fit the classifier and label the testing split
        clf = classifier.fit(X_train, y_train)

        preditctions = clf.predict(X_test)

        #Caculate the F1 score and store it
        results[i] = format(f1_score(y_test, preditctions, average='macro'), ".3f")

    p = pickle.dumps(clf)
    return (format(sum(map(float, results)) / len(results), ".3f"), format(sys.getsizeof(p) / 1000, ".3f"))

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

originalDataset = os.path.join(dirname, "Data/ESC-50-Folds-Compressed-4s")
run_grid_search(originalDataset, "-Augmented-MFCC", [256, 512, 1024, 2048, 4096], [10, 15, 20, 25, 30], [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2], "U:/GDP/ML Testing/Low-Power-Sound-Recognition-Classification/Output/grid_search-ESC-50-MFCC-Fixed-Hop-Length.csv")