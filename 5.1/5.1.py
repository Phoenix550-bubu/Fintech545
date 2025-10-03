import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

def normal_simulation_pd_input(input_cov):
    cov_matrix = input_cov.values
    n_vars = cov_matrix.shape[0]

    n_simulations = 100000
    mean_vector = np.zeros(n_vars)

    np.random.seed(42)
    simulated_data = multivariate_normal.rvs(
        mean=mean_vector,
        cov=cov_matrix,
        size=n_simulations
    )

    output_cov = np.cov(simulated_data.T)

    result_cov = pd.DataFrame(output_cov,columns=input_cov.columns)

    return result_cov

def main():
    input_cov = pd.read_csv('test5_1.csv')

    result_cov = normal_simulation_pd_input(input_cov)

    print(','.join(result_cov.columns))

    for _, row in result_cov.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()