import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import DictionaryLearning
import FeatureExtractor as fe

(sampleRate, audioData, frameSize) = fe.get_wav_data("U:\GDP\ML Testing\Low-Power-Sound-Recognition-Classification\Data\ESC-50\chainsaw\\1-19898-A-41.wav", 0.1)

#FIX THIS / MAKE IT NOT STOLEN
def gabor_atom(
    N: int,
    w: float,
    s: float,
    u: int,
    theta: float = 0
):
    return np.sqrt(1 / s) * np.exp(-np.pi*np.square(np.arange(N)-u)/(s*s)) \
        * np.cos(2*np.pi*w*(np.arange(N)-u) - theta)

def gabor_basis(
    N: int,
    i_vals: list = range(1, 36),
    p_vals: list = range(1, 9),
    u_step: int = 64,
    dtype: type = np.float32
):
    """
    Gabor basis.
    """
    K = 0.5 * i_vals[-1]**(-2.6)

    a_freqs = [K*i**2.6 for i in i_vals]
    scales = [2**p for p in p_vals]
    time_shifts = range(0, N, u_step)

    dictionary = []

    for i in range(len(a_freqs)):
        for j in range(len(scales)):
            for k in range(len(time_shifts)):
                cunt = gabor_atom(N, a_freqs[i], scales[j], time_shifts[k])
                dictionary.append(gabor_atom(N, a_freqs[i], scales[j],
                    time_shifts[k]))

    return np.array(dictionary, dtype)

n_components, n_features = 512, 100
n_nonzero_coefs = 17

# generate the data

# y = Xw
# |x|_0 = n_nonzero_coefs

y = np.abs(np.fft.rfft(audioData[:int(sampleRate * 1.5)]))
X = gabor_basis(1000)

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, normalize=False)
omp.fit(X, y)
coef = omp.coef_
(idx_r,) = coef.nonzero()
plt.stem(idx_r, coef[idx_r], use_line_collection=True)

plt.show()