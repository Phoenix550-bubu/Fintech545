import pandas as pd
import numpy as np
from numpy import inf, copy
from numpy.linalg import norm

def proj_psd(A):
    d, v = np.linalg.eigh(A)
    A = (v * np.maximum(d, 0)).dot(v.T)
    A = (A + A.T) / 2
    return A

def higham_nearcorr(A, tol=None, max_iterations=100, weights=None):
    if isinstance(A, pd.DataFrame):
        A = A.values

    eps = np.spacing(1)
    if not np.all((np.transpose(A) == A)):
        raise ValueError('Input Matrix is not symmetric')

    if tol is None:
        tol = eps * np.shape(A)[0] * np.array([1, 1])

    if weights is None:
        weights = np.ones(np.shape(A)[0])

    X = copy(A)
    Y = copy(A)
    ds = np.zeros(np.shape(A))
    rel_diffY = inf
    rel_diffX = inf
    rel_diffXY = inf

    Whalf = np.sqrt(np.outer(weights, weights))

    iteration = 0
    while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
        iteration += 1
        if iteration > max_iterations:
            break

        Xold = copy(X)
        R = X - ds
        R_wtd = Whalf * R
        X = proj_psd(R_wtd)
        X = X / Whalf
        ds = X - R
        Yold = copy(Y)
        Y = copy(X)
        np.fill_diagonal(Y, 1)
        normY = norm(Y, 'fro')
        rel_diffX = norm(X - Xold, 'fro') / norm(X, 'fro')
        rel_diffY = norm(Y - Yold, 'fro') / normY
        rel_diffXY = norm(Y - X, 'fro') / normY

        X = copy(Y)

    return X

def higham_psd(a):
    if isinstance(a, pd.DataFrame):
        a = a.values

    out = a.copy()

    invSD = None
    diag_elements = np.diag(out)

    if not np.allclose(diag_elements, 1.0, rtol=1e-8):
        invSD = np.diag(1.0 / np.sqrt(diag_elements))
        out = invSD @ out @ invSD

    out = (out + out.T) / 2

    out = higham_nearcorr(out)

    if invSD is not None:
        invSD_back = np.diag(1.0 / np.diag(invSD))
        out = invSD_back @ out @ invSD_back

    return out

def main():
    data = pd.read_csv('testout_1.3.csv')

    higham_matrix = higham_psd(data)

    print(','.join(data.columns))

    for row in higham_matrix:
        print(','.join(map(str, row)))

if __name__ == "__main__":
    main()