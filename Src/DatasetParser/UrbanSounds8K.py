import os
import csv

def read_csv_label_data(csvFileName, dir) :
    with open(csvFileName, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            relocate_file(dir + "/fold" + row['fold'], row['slice_file_name'], row['class'])

def relocate_file(dir, filename, label) :
    os.makedirs(dir + "/" + label, exist_ok=True)
    os.rename(dir + "/" + filename, dir + "/" + label + "/" + filename)

read_csv_label_data("../../Data/UrbanSounds8k/UrbanSound8K.csv", "../../Data/UrbanSounds8k")