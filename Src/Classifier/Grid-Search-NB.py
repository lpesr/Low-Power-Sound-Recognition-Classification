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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import pandas as pd
import os
import pickle
import sys
import time

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.join(dirname, 'Src/FeatureExtractor'))
import DataPrep as dp

param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

numFolds = 5
#(dataFolds, labelFolds) = dp.prepare_input_data_with_folds(os.path.join(dirname, "./Data/UrbanSounds8k"), ["drilling", "gun_shot", "siren", "children_playing", "car_horn"], 0.05, 0.5, 4)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"
(dataFolds, labelFolds) = dp.prepare_input_data_with_folds(os.path.join(dirname, "./Data/ESC-50-Folds"), ["glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"], 0.01, 1, 1, numFolds)#4000)#, "jackhammer", "siren", "dog_bark"], 1500) #"glass_breaking", "siren", "hand_saw", "vacuum_cleaner", "crackling_fire"

dirname = os.path.dirname(__file__)

#Define all of the classifiers to test
classifier = GaussianNB()

#Normalize the data
scaler = MinMaxScaler()
for k in range(0, len(dataFolds)):
    dataFolds[k] = scaler.fit_transform(dataFolds[k])

nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
nbModel_grid.fit([item for sublist in dataFolds for item in sublist], [item for sublist in labelFolds for item in sublist])
print(nbModel_grid.best_estimator_)
