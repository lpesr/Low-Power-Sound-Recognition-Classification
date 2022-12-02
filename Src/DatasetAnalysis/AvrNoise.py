import scipy.io.wavfile as wavfile
import numpy
import os.path
import scipy.stats as stats

def snr(file):
  if (os.path.isfile(file)):
    data = wavfile.read(file)[1]
    singleChannel = data
    try:
      singleChannel = numpy.sum(data, axis=1)
    except:
      # was mono after all
      pass
      
    norm = singleChannel / (max(numpy.amax(singleChannel), -1 * numpy.amin(singleChannel)))
    return stats.signaltonoise(norm)