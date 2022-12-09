import os
import csv

def read_csv_label_data(csvFileName, dir) :
    """Seperate the Urbansound8k files into sub dirs with folds and then a sub dir for the 
    respective labels. This is done by reading the Urbansound8k csv files that puts each
    file into folds and labels.
    Prams:
        csvFileName = Path to the Urbansound8k csv file
        dir = Dir of the files
    """
    with open(csvFileName, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            relocate_file(dir + "/fold" + row['fold'], row['slice_file_name'], row['class'])

def relocate_file(dir, filename, label) :
    os.makedirs(dir + "/" + label, exist_ok=True)
    os.rename(dir + "/" + filename, dir + "/" + label + "/" + filename)

read_csv_label_data("../../Data/UrbanSounds8k/UrbanSound8K.csv", "../../Data/UrbanSounds8k")