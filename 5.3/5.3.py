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

def normal_simulation_nonpsd_input(input_cov):
    fixed_cov = near_psd(input_cov, epsilon=0.0)

    cov_matrix = fixed_cov
    n_vars = cov_matrix.shape[0]

    n_simulations = 100000
    mean_vector = np.zeros(n_vars)

    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    eigenvals = np.maximum(eigenvals, 0)

    np.random.seed(30)
    standard_samples = np.random.normal(0, 1, (n_simulations, n_vars))

    sqrt_eigenvals = np.sqrt(eigenvals)
    transform_matrix = eigenvecs @ np.diag(sqrt_eigenvals)

    simulated_data = mean_vector + standard_samples @ transform_matrix.T

    output_cov = np.cov(simulated_data.T)

    result_cov = pd.DataFrame(output_cov, columns=input_cov.columns)

    return result_cov

def main():
    input_cov = pd.read_csv('test5_3.csv')

    result_cov = normal_simulation_nonpsd_input(input_cov)

    print(','.join(result_cov.columns))

    for _, row in result_cov.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()
