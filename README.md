# Low-Power-Sound-Recognition-Classification

This project investigates the appropriability of using machine learning models on an ultra-low powered system.

The system requirements are power consumption in the micro-Watt range, along with <40KB of ram. 
The goal is to be able to identify sounds picked up by the microphone on the microcontroller.
Two approaches will be looked at, identifying environmental sounds as well as voice detection.

## Approaches

### Machine Learning Models

The following models have been evaluated:
- "Nearest Neighbors",
- "Linear SVM",
- "RBF SVM",
- "Sigmoid SVM",
- "Decision Tree",
- "Random Forest",
- "Neural Net",
- "Naive Bayes"

### Feature Extraction

The feature vector needs to be very small so deep learning on a spectrogram is not a valid approach.
However, by taking spectral centroids the dimensionality of the feature vector is greatly decreased.
Zero Crossing Rate (ZCR) is also being evaluated as a small feature vector that generates a lot of information.

## Datasets

There are two datasets that are being used. First of all the ESC-50 dataset, this is a dataset that contaions
50 classes of high qaulity 10 second audio clips. The biggest drawback is that there are only 40 WAV files per
class. The second dataset that is being used is the UrbanSounds8K dataset. This dataset has over 8000 audio files
for 10 classes that are each a maximum of 5 seconds long.

### ESC-50

The ESC-50 is small enough to fit onto the git and can be found in the Data directory.

### UrbanSounds8K

The UrbanSounds8K dataset is very large and for convenience needs to be downloaded separately and pared using the
dataset pasing python scrpits.

https://urbansounddataset.weebly.com/urbansound8k.html

## Setup

requierments.txt contains a lit of python modules that can be installed with the 
following command:

```BASH
pip install -r Requirements.txt 
```

### Requirements:
- numpy
- sklearn
- pandas
- matplotlib
- openpyxl
- pickle
- time
- sys
- librosa

## Dataprep

Once the UrbanSounds8K dataset is downloaded and extracted into the Data folder run the data parser script.

```BASH
python "./Src/DatasetParser/UrbanSounds8K.py"
```

## Testing

To test the machine learning models run the desired scrpit from the Src/Classifier directory.

- TestClassifier.py: Tests the classifiers on a single set of paramaters,
- Test-ClassifierVsClasses.py: Tests the classifiers on 2 to 8 classes, outputting in tablur format.