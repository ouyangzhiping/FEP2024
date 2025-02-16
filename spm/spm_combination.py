import numpy as np

def spm_combinations(Nu):
    """
    Returns a matrix of all combinations of Nu.
    
    Parameters:
    Nu (list or array): Vector of dimensions
    
    Returns:
    np.ndarray: Combinations of indices
    """
    Nf = len(Nu)
    U = np.zeros((np.prod(Nu), Nf), dtype=int)
    
    for f in range(Nf):
        k = []
        for j in range(Nf):
            if j == f:
                k.append(np.arange(1, Nu[j] + 1))
            else:
                k.append(np.ones(Nu[j], dtype=int))
        
        u = np.array([1])
        for i in range(Nf):
            u = np.kron(k[i], u)
        
        # accumulate
        U[:, f] = u.flatten()
    
    return U