import pandas as pd
import numpy as np

def exponential_weights(n, lambda_param):
    weights = np.zeros(n)
    total_weight = 0.0

    for i in range(n):
        weights[i] = (1 - lambda_param) * (lambda_param ** (i + 1))
        total_weight += weights[i]

    weights = weights / total_weight

    return weights

def exponential_weighted_covariance(data, lambda_param=0.97):
    X = data.values
    n, m = X.shape

    weights = exponential_weights(n, lambda_param)
    weights = weights[::-1]

    weighted_means = np.zeros(m)
    for j in range(m):
        weighted_means[j] = np.sum(weights * X[:, j])

    cov_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            weighted_cov = 0.0
            for t in range(n):
                weighted_cov += weights[t] * (X[t, i] - weighted_means[i]) * (X[t, j] - weighted_means[j])

            cov_matrix[i, j] = weighted_cov

    return cov_matrix

def main():
    data = pd.read_csv('test2.csv')

    ewm_cov = exponential_weighted_covariance(data, lambda_param=0.97)

    print(','.join(data.columns))

    for row in ewm_cov:
        print(','.join(map(str, row)))

if __name__ == "__main__":
    main()