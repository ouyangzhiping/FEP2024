import numpy as np

def spm_log(A):
    """
    Log of numeric array plus a small constant.
    """
    return np.log(A + 1e-16)

def spm_norm(A):
    """
    Normalization of a probability transition matrix (columns).
    """
    A = A / np.sum(A, axis=0, keepdims=True)
    A[np.isnan(A)] = 1 / A.shape[0]
    return A

def spm_wnorm(A):
    """
    This function normalizes the input matrix A.
    It adds a small constant to A, then uses broadcasting to subtract the inverse of each column
    entry from the inverse of the sum of the columns and then divides by 2.
    """
    A = A + np.exp(-16)
    A = (1.0 / np.sum(A, axis=0) - 1.0 / A) / 2.0
    return A

def spm_ind2sub(siz, ndx):
    """
    Subscripts from linear index.
    """
    n = len(siz)
    k = np.cumprod([1] + list(siz[:-1]))
    sub = np.zeros(n, dtype=int)
    for i in range(n-1, -1, -1):
        vi = (ndx - 1) % k[i] + 1
        vj = (ndx - vi) // k[i] + 1
        sub[i] = vj
        ndx = vi
    return sub