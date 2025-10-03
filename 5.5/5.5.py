import pandas as pd
import numpy as np

def pca_simulation(input_cov, n_simulations=100000, explained_variance=0.99):
    columns = input_cov.columns if isinstance(input_cov, pd.DataFrame) else None
    cov_matrix = input_cov.values if isinstance(input_cov, pd.DataFrame) else input_cov

    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    total_variance = eigenvals.sum()

    positive_idx = eigenvals >= 1e-8
    eigenvals = eigenvals[positive_idx]
    eigenvecs = eigenvecs[:, positive_idx]

    cumulative_variance = np.cumsum(eigenvals)
    variance_ratio = cumulative_variance / total_variance
    n_components = np.argmax(variance_ratio >= explained_variance) + 1

    selected_eigenvals = eigenvals[:n_components]
    selected_eigenvecs = eigenvecs[:, :n_components]

    transform_matrix = selected_eigenvecs @ np.diag(np.sqrt(selected_eigenvals))

    np.random.seed(42)
    standard_samples = np.random.normal(0, 1, (n_components, n_simulations))

    simulated_data = (transform_matrix @ standard_samples).T

    output_cov = np.cov(simulated_data.T)

    result_cov = pd.DataFrame(output_cov, columns=input_cov.columns)

    return result_cov

def main():
    input_cov = pd.read_csv('test5_2.csv')

    result_cov = pca_simulation(input_cov, n_simulations=100000, explained_variance=0.99)

    print(','.join(result_cov.columns))

    for _, row in result_cov.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()
