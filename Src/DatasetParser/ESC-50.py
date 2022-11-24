import os
import csv

def read_csv_label_data(csvFileName, dir) :
    with open(csvFileName, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            relocate_file_with_fold(dir, row['filename'], row['category'], row['fold'])

def relocate_file(dir, filename, label) :
    os.makedirs(dir + "/" + label, exist_ok=True)
    os.rename(dir + "/audio/" + filename, dir + "/" + label + "/" + filename)

def relocate_file_with_fold(dir, filename, label, fold) :
    os.makedirs(dir + "/fold" + fold, exist_ok=True)
    os.makedirs(dir + "/fold" + fold + "/" + label, exist_ok=True)
    os.rename(dir + "/" + filename, dir + "/fold" + fold + "/" + label + "/" + filename)

read_csv_label_data("U:/GDP/ML Testing/Low-Power-Sound-Recognition-Classification/Data/ESC-50-master/meta/esc50.csv", "U:/GDP/ML Testing/Low-Power-Sound-Recognition-Classification/Data/ESC-50-master/audio")