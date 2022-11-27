import os
import sys
import numpy as np

numOfClasses = 8
softMemoryThreshold = 40
hardMemoryThreshold = 120

classifiers = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Sigmoid SVM", "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes"]
#classifiers = ["Decision Tree", "Random Forest", "Neural Net", "Naive Bayes"]

bestValue = [0] * (numOfClasses - 1) + [sys.float_info.max] * (numOfClasses - 1) * 2

csvFileName = "U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Src\ResultsFormatter\input.txt"
outputFile = "U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Src\ResultsFormatter\output.txt"

reader = np.genfromtxt(csvFileName, delimiter='&', comments="\\\\") 
for row in range(0, len(reader)):
    for col in range(1, len(reader[row])):
        if (col < numOfClasses and reader[row][col] > bestValue[col - 1]) or (col >= numOfClasses and reader[row][col] < bestValue[col - 1]):
            bestValue[col - 1] = reader[row][col]

os.remove(outputFile)
f = open(outputFile, "a")
for row in range(0, len(reader)):
    f.write(classifiers[row])
    for col in range(1, len(reader[row])):
        if reader[row][col] > hardMemoryThreshold and col >= numOfClasses and col < numOfClasses * 2 - 1:
            f.write(" & \hlred{" + format(reader[row][col], ".3f") + "}")
        elif reader[row][col] > softMemoryThreshold and col >= numOfClasses and col < numOfClasses * 2 - 1:
            f.write(" & \hlyellow{" + format(reader[row][col], ".3f") + "}")
        elif bestValue[col - 1] == reader[row][col]:
            f.write(" & \hllime{" + format(reader[row][col], ".3f") + "}")
        else:
            f.write(" & " + format(reader[row][col], ".3f"))
    f.write("\\\\\n")
f.close()
