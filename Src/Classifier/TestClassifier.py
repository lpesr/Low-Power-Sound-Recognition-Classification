import FeatureExtractor as fe
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.svm import SVC
import pandas as pd
import os

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

data = prepare_input_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data", ["airplane", "breathing"], 500)

dirname = os.path.dirname(__file__)

#Define all of the classifiers to test
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", gamma="auto", C=10),
    SVC(kernel="rbf", gamma="auto", decision_function_shape='ovo', C=1000),
    SVC(kernel="sigmoid", gamma="auto", C=10),
    DecisionTreeClassifier(max_depth=100),
    RandomForestClassifier(max_depth=100, n_estimators=20, max_features=5),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(learning_rate=0.5),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    #GaussianProcessClassifier(1.1 * RBF(1.0), random_state=0, multi_class="one_vs_one"),
]

numKFoldSplits = 10
numKFoldRep = 2

#Create a results matrix
i, j = numKFoldSplits * numKFoldRep, len(classifiers)
results = [[0 for x in range(i)] for y in range(j)] 

#Iterate over all of the classifiers
for j in range(len(classifiers)):
    i = 0

    #K cross validate the data (there will be an equal number of both classes to train on)
    #This is because the data was split and then combined earlier
    kf = RepeatedKFold(n_splits=numKFoldSplits, n_repeats=numKFoldRep)
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        #Fit the classifier and label the testing split
        clf = classifiers[j].fit(X_train, y_train)
        preditctions = clf.predict(X_test)

        #Caculate the F1 score and store it
        results[j][i] = format(f1_score(y_test, preditctions, average='macro'), ".3f")

        #Caculate the acuraccy and store it
        #correct = len([i for i,(predition, actual) in enumerate(zip(preditctions, y_test)) if predition == actual])
        #results[j][i] = correct / len(preditctions)
        
        i += 1

    print((names[j], results[j], format(sum(map(float, results[j][:])) / len(results[j]), ".3f")))