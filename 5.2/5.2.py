import pandas as pd
import numpy as np

def normal_simulation_psd_input(input_cov):
    cov_matrix = input_cov.values
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
    input_cov = pd.read_csv('test5_2.csv')

    result_cov = normal_simulation_psd_input(input_cov)

    print(','.join(result_cov.columns))

    for _, row in result_cov.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()