import numpy as np

def spm_iwft(C, k, n):
    """
    Inverse windowed Fourier transform - continuous synthesis
    :param C: coefficients (complex)
    :param k: Frequencies (cycles per window)
    :param n: window length
    :return: 1-D time-series
    """
    # window function (Hanning)
    N = C.shape[1]
    s = np.zeros(N)
    C = np.conj(C)

    # spectral density
    for i in range(len(k)):
        W = np.exp(-1j * (2 * np.pi * k[i] * np.arange(N) / n))
        w = W * C[i, :]
        s += np.real(w)
    
    return s