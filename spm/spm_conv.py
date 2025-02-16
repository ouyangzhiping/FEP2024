import numpy as np
from scipy.ndimage import convolve1d

def spm_conv(X, sx, sy=None):
    """
    Gaussian convolution
    Parameters:
    X  - matrix
    sx - kernel width (FWHM) in pixels
    sy - optional non-isomorphic smoothing
    """
    if sy is None:
        sy = sx
    sx = abs(sx)
    sy = abs(sy)
    lx, ly = X.shape

    # FWHM -> sigma
    sx = sx / np.sqrt(8 * np.log(2)) + np.finfo(float).eps
    sy = sy / np.sqrt(8 * np.log(2)) + np.finfo(float).eps

    # kernels
    Ex = min(int(6 * sx), lx)
    x = np.arange(-Ex, Ex + 1)
    kx = np.exp(-x**2 / (2 * sx**2))
    kx = kx / np.sum(kx)
    Ey = min(int(6 * sy), ly)
    y = np.arange(-Ey, Ey + 1)
    ky = np.exp(-y**2 / (2 * sy**2))
    ky = ky / np.sum(ky)

    # convolve
    if lx > 1 and len(kx) > 1:
        for i in range(ly):
            u = X[:, i]
            v = np.concatenate((np.flipud(u[:Ex]), u, np.flipud(u[-Ex:])))
            X[:, i] = convolve1d(v, kx, mode='constant', cval=0.0)[Ex:-Ex]

    if ly > 1 and len(ky) > 1:
        for i in range(lx):
            u = X[i, :]
            v = np.concatenate((np.fliplr([u[:Ey]])[0], u, np.fliplr([u[-Ey:]])[0]))
            X[i, :] = convolve1d(v, ky, mode='constant', cval=0.0)[Ey:-Ey]

    return X