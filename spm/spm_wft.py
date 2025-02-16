import numpy as np

def spm_wft(s, k, n):
    """
    Windowed Fourier wavelet transform (time-frequency analysis)
    
    Parameters:
    s (ndarray): (t X n) time-series
    k (ndarray): Frequencies (cycles per window)
    n (int): window length
    
    Returns:
    ndarray: (w X t X n) coefficients (complex)
    """
    
    # Window function (Hanning)
    T, N = s.shape
    n = round(n)
    h = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1)))
    h = h / np.sum(h)
    C = np.zeros((len(k), T, N), dtype=complex)
    
    # Spectral density
    for i in range(len(k)):
        W = np.exp(-1j * (2 * np.pi * k[i] * np.arange(T) / n))
        for j in range(N):
            w = np.convolve(s[:, j] * W, h, mode='same')
            C[i, :, j] = w
    
    return C