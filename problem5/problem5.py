import pandas as pd
import numpy as np
from numpy import inf, copy
from numpy.linalg import norm
import warnings
warnings.filterwarnings('ignore')


def missing_cov(x, skip_miss=True, fun=np.cov):
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)

    n, m = x.shape
    n_miss = x.isnull().sum()

    if n_miss.sum() == 0:
        return fun(x.T)

    if skip_miss:
        complete_rows = x.dropna()
        if len(complete_rows) == 0:
            raise ValueError("No complete rows found")
        return fun(complete_rows.T)
    else:
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


def check_matrix_definiteness(matrix):
    eigenvals = np.linalg.eigvalsh(matrix)

    min_eigenval = np.min(eigenvals)
    max_eigenval = np.max(eigenvals)

    tol = 1e-8

    if min_eigenval > tol:
        return "Positive Definite (PD)", eigenvals
    elif min_eigenval >= -tol:
        return "Positive Semi-Definite (PSD)", eigenvals
    else:
        return "Non-Definite", eigenvals


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


def pca_variance_explained(cov_matrix):
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    total_variance = eigenvals.sum()

    variance_explained = eigenvals / total_variance
    cumulative_variance = np.cumsum(variance_explained)

    return eigenvals, variance_explained, cumulative_variance


def main():
    df = pd.read_csv('problem5.csv')

    print("Part a")

    pairwise_cov = missing_cov(df, skip_miss=False)

    print(f"{'':>8}", end="")
    for col in df.columns:
        print(f"{col:>15}", end="")
    print()

    for i, row_name in enumerate(df.columns):
        print(f"{row_name:>8}", end="")
        for j in range(len(df.columns)):
            print(f"{pairwise_cov[i, j]:>15.10f}", end="")
        print()


    print("-" * 80)
    print("Part b")

    definiteness, eigenvals_original = check_matrix_definiteness(pairwise_cov)

    eigenvals_sorted = np.sort(eigenvals_original)[::-1]
    for i, eigenval in enumerate(eigenvals_sorted):
        print(f"λ{i+1}: {eigenval:>15.10f}")



    if definiteness == "Non-Definite":
        print("-" * 80)
        print("Part c")

        fixed_cov = higham_psd(pairwise_cov)

        print(f"{'':>8}", end="")
        for col in df.columns:
            print(f"{col:>15}", end="")
        print()

        for i, row_name in enumerate(df.columns):
            print(f"{row_name:>8}", end="")
            for j in range(len(df.columns)):
                print(f"{fixed_cov[i, j]:>15.10f}", end="")
            print()

        cov_for_pca = fixed_cov
    else:
        cov_for_pca = pairwise_cov


    print("-" * 80)
    print("Part d")

    eigenvals_pca, variance_explained, cumulative_variance = pca_variance_explained(cov_for_pca)

    print(f"{'PC':>5} {'Eigenvalue':>15} {'Variance%':>15} {'Cumulative%':>15}")

    for i in range(len(eigenvals_pca)):
        print(f"{'PC' + str(i+1):>5} {eigenvals_pca[i]:>15.10f} {variance_explained[i]*100:>14.6f}% {cumulative_variance[i]*100:>14.6f}%")


if __name__ == "__main__":
    main()

