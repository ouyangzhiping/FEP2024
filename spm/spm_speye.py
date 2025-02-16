import numpy as np
from scipy.sparse import diags, csr_matrix

def spm_speye(m, n=None, k=0, c=0):
    """
    Sparse leading diagonal matrix

    Returns an m x n matrix with ones along the k-th leading diagonal. If
    called with an optional fourth argument c = 1, a wraparound sparse matrix
    is returned. If c = 2, then empty rows or columns are filled in on the
    leading diagonal.
    """
    if n is None:
        n = m

    # leading diagonal matrix
    D = diags([1] * m, k, shape=(m, n), format='csr')

    # add wraparound if necessary
    if c == 1:
        if k < 0:
            D = D + spm_speye(m, n, min(n, m) + k)
        elif k > 0:
            D = D + spm_speye(m, n, k - min(n, m))
    elif c == 2:
        i = np.where(~D.toarray().any(axis=0))[0]
        D = D + csr_matrix((np.ones(len(i)), (i, i)), shape=(m, n))

    return D

# Example usage
# D = spm_speye(5, 5, 0, 2)
# print(D.toarray())