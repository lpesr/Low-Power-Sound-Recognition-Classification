import deeplake
ds = deeplake.load("hub://activeloop/esc50")




from pyAudioAnalysis import audioTrainTest as aT

aT.extract_features_and_train(["../../Data/ESC-50/101 - Dog","../../Data/ESC-50/102 - Rooster"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
aT.file_classification("../../Data/ESC-50/103 - Pig\1-208757-A.ogg", "svmSMtemp","svm")