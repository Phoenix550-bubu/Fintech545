import pandas as pd
import numpy as np

def chol_psd(a):
    if isinstance(a, pd.DataFrame):
        a = a.values

    n = a.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        temp = a[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir

    return root

def main():
    data = pd.read_csv('testout_3.1.csv')

    chol_matrix = chol_psd(data)

    print(','.join(data.columns))

    for row in chol_matrix:
        print(','.join(map(str, row)))

if __name__ == "__main__":
    main()