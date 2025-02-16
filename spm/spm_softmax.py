import numpy as np

def spm_softmax(x, k=1):
    """
    Softmax (e.g., neural transfer) function over columns

    Parameters:
    x - numeric array
    k - precision, sensitivity or inverse temperature (default k = 1)

    Returns:
    y - softmax values
    """
    # Apply precision, sensitivity or inverse temperature
    if k != 1:
        x = k * x

    # If input has less than 2 rows, return an array of ones
    if x.shape[0] < 2:
        return np.ones_like(x)

    # Exponentiate and normalize
    x = np.exp(x - np.max(x, axis=0))
    y = x / np.sum(x, axis=0)
    
    return y