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

def exponential_weighted_covariance(data, lambda_param=0.94):
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

def exponential_weighted_correlation(data, lambda_param=0.94):
    cov_matrix = exponential_weighted_covariance(data, lambda_param)

    std_devs = np.sqrt(np.diag(cov_matrix))

    m = cov_matrix.shape[0]
    corr_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if std_devs[i] > 0 and std_devs[j] > 0:
                corr_matrix[i, j] = cov_matrix[i, j] / (std_devs[i] * std_devs[j])
            else:
                corr_matrix[i, j] = 0.0

    return corr_matrix

def main():
    data = pd.read_csv('test2.csv')

    ewm_corr = exponential_weighted_correlation(data, lambda_param=0.94)

    print(','.join(data.columns))

    for row in ewm_corr:
        print(','.join(map(str, row)))

if __name__ == "__main__":
    main()