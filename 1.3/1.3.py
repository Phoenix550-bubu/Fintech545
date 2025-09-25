import pandas as pd
import numpy as np
from typing import Optional, Callable

def missing_cov(x, skip_miss=True, fun=np.cov):
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)

    n, m = x.shape

    n_miss = x.isnull().sum()

    if n_miss.sum() == 0:
        return fun(x.T)

    if skip_miss:
        # Skip missing
        complete_rows = x.dropna()
        if len(complete_rows) == 0:
            raise ValueError("No complete rows found")
        return fun(complete_rows.T)
    else:
        # Pairwise
        out = np.zeros((m, m))

        for i in range(m):
            for j in range(i + 1):
                valid_mask = x.iloc[:, i].notna() & x.iloc[:, j].notna()
                valid_data = x.loc[valid_mask, [x.columns[i], x.columns[j]]]

                if len(valid_data) < 2:
                    out[i, j] = 0.0
                else:
                    cov_matrix = fun(valid_data.T)
                    if cov_matrix.ndim == 0:
                        out[i, j] = cov_matrix
                    else:
                        out[i, j] = cov_matrix[0, 1] if i != j else cov_matrix[0, 0]

                if i != j:
                    out[j, i] = out[i, j]

        return out

def main():
    data = pd.read_csv('test1.csv')

    cov_pairwise = missing_cov(data, skip_miss=False)

    print(','.join(data.columns))

    for row in cov_pairwise:
        print(','.join(map(str, row)))

if __name__ == "__main__":
    main()