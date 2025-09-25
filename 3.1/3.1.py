import pandas as pd
import numpy as np

def near_psd(a, epsilon=0.0):
    if isinstance(a, pd.DataFrame):
        a = a.values

    n = a.shape[0]
    out = a.copy()

    invSD = None
    diag_elements = np.diag(out)

    if not np.allclose(diag_elements, 1.0, rtol=1e-8):
        invSD = np.diag(1.0 / np.sqrt(diag_elements))
        out = invSD @ out @ invSD

    eigenvals, eigenvecs = np.linalg.eigh(out)
    eigenvals = np.maximum(eigenvals, epsilon)

    T = 1.0 / (eigenvecs * eigenvecs @ eigenvals)
    T = np.diag(np.sqrt(T))
    L = np.diag(np.sqrt(eigenvals))
    B = T @ eigenvecs @ L
    out = B @ B.T

    if invSD is not None:
        invSD_back = np.diag(1.0 / np.diag(invSD))
        out = invSD_back @ out @ invSD_back

    return out

def main():
    data = pd.read_csv('testout_1.3.csv')

    near_psd_matrix = near_psd(data)

    print(','.join(data.columns))

    for row in near_psd_matrix:
        print(','.join(map(str, row)))

if __name__ == "__main__":
    main()