import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import pandas as pd
import os

dirname = os.path.dirname(__file__)

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Sigmoid SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "LDA",
    #"Gaussian Process",
]

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

#Read the training data
data = pd.read_csv(os.path.join(dirname, '../../Data/TrainingData.txt'), header=None)
data = np.array(data)
#Split the traning data into the two classes, then combine the data so classes are in the order {0,1,0,1,...,0,1}
data[::2], data[1::2] = [data[(data[:,24]==0)], data[(data[:,24]==1)]]

#Split the data from the class identifer
y = data[:,24]
X = np.delete(data, 24, 1)

#Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#Create a results matrix
i, j = numKFoldSplits * numKFoldRep, len(classifiers)
results = [[0 for x in range(i)] for y in range(j)] 

print("  [Algorithm]-----------------------------------------------------------------------------------------[Results]------------------------------------------------------------------------------------------[Average]  ")

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