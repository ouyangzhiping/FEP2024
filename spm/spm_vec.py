import numpy as np

def spm_vec(X, *args):
    """
    Vectorise a numeric, list or dictionary array.
    FORMAT [vX] = spm_vec(X)
    X  - numeric, list or dictionary array[s]
    vX - vec(X)
    
    See spm_unvec
    
    e.g.:
    spm_vec([np.eye(2), 3]) = [1, 0, 0, 1, 3]
    """
    
    # Initialise X and vX
    if args:
        X = [X] + list(args)
    
    # Vectorise numerical arrays
    if isinstance(X, (np.ndarray, int, float)):
        vX = np.array(X).flatten()
    
    # Vectorise logical arrays
    elif isinstance(X, (np.bool_, bool)):
        vX = np.array(X).flatten()
    
    # Vectorise dictionary into list arrays
    elif isinstance(X, dict):
        vX = []
        for key in X:
            vX = np.concatenate((vX, spm_vec(X[key])))
    
    # Vectorise lists into numerical arrays
    elif isinstance(X, list):
        vX = []
        for item in X:
            vX = np.concatenate((vX, spm_vec(item)))
    
    else:
        vX = np.array([])
    
    return vX