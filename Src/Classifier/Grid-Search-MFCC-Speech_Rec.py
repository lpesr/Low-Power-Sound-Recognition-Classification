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

def run_grid_search(originalData, augDataset, numFFTs, numMFCCs, fileLengths, hopDivisions, outputFile):
    output = open(outputFile, "a")

    for hopDivision in hopDivisions:
        output.write("hopDivisions= " + str(hopDivision) + ",\n\n")
        for numMFCC in numMFCCs:
            output.write("numMFCCs= " + str(numMFCC) + ",\n")

            output.write("[]")
            for fileLength in fileLengths:
                output.write("," + str(fileLength))
            output.write("\n")

            for numFFT in numFFTs:
                output.write(str(numFFT) + ":")
                for fileLength in fileLengths:
                    output.write(",")
                    result = test_dataset(originalData, augDataset, numFFT, numMFCC, int(numFFT / hopDivision), fileLength)
                    output.write(str(result[0]) + " | " + str(result[1]))
                output.write("\n")
                output.flush()

            output.write("\n")
        output.write("\n\n")
    output.close()
            
def test_dataset(originalDataset, augmentedDataset, numFFT, numMFCC, hopLength, fileLength):
    (X, Y) = dp.prepare_input_data(originalDataset, ["yes", "no", "on", "off"], 0, fileLength, 4, numFft=numFFT, numMFCC=numMFCC, hopLength=hopLength, augmentation=augmentedDataset)
    X = np.array(X)
    Y = np.array(Y)

    numKFoldSplits = 10
    numKFoldRep = 2

    #Create a results matrix
    i = 0
    results = [0 for x in range(numKFoldSplits * numKFoldRep)]

    kf = RepeatedKFold(n_splits=numKFoldSplits, n_repeats=numKFoldRep)
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]

        #Fit the classifier and label the testing split
        clf = classifier.fit(X_train, y_train)

        preditctions = clf.predict(X_test)

        #Caculate the F1 score and store it
        results[i] = format(f1_score(y_test, preditctions, average='macro'), ".3f")
        i += 1

    p = pickle.dumps(clf)
    return (format(sum(map(float, results)) / len(results), ".3f"), format(sys.getsizeof(p) / 1000, ".3f"))

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

originalDataset = os.path.join(dirname, "Data/Speech-Commands")
run_grid_search(originalDataset, (0, 0, 0), [256, 512, 1024, 2048, 4096], [10, 15, 20, 25, 30], [0.25, 0.5, 0.75, 1, 1.25, 1.5], [1], "U:/GDP/ML Testing/Low-Power-Sound-Recognition-Classification/Output/grid_search-Speech-Recognition-MFCC.csv")