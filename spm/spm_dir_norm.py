import numpy as np

def spm_dir_norm(A):
    """
    Normalisation of a (Dirichlet) conditional probability matrix
    Parameters:
    A - (Dirichlet) parameters of a conditional probability matrix

    Returns:
    A - conditional probability matrix
    """
    A = A / np.sum(A, axis=1, keepdims=True)
    A[np.isnan(A)] = 1 / A.shape[0]
    return A