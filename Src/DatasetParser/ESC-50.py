import os
import csv

def read_csv_label_data(csvFileName, dir) :
    with open(csvFileName, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            relocate_file(dir, row['filename'], row['category'])

def relocate_file(dir, filename, label) :
    os.makedirs(dir + "\\" + label, exist_ok=True)
    os.rename(dir + "\\audio\\" + filename, dir + "\\" + label + "\\" + filename)

read_csv_label_data("..\\..\\Data\\ESC-50\\esc50.csv", "..\\..\\Data\\ESC-50")