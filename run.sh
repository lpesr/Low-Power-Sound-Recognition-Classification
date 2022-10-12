#!/bin/bash

#Install requirements
pip install -r requirements.txt 

#Label the input list
python "./Src/Classifier/Classifier.py"
#Solve for optimal solutions
python "./Src/Task Planer/TaskSolver.py"