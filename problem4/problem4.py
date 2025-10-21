import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


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


def exponential_weighted_variances(data, lambda_param=0.97):
    cov_matrix = exponential_weighted_covariance(data, lambda_param)
    variances = np.diag(cov_matrix)
    return variances


def combine_correlation_variance_to_covariance(corr_matrix, variances):
    std_devs = np.sqrt(variances)
    m = len(variances)
    cov_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if i == j:
                cov_matrix[i, j] = variances[i]
            else:
                cov_matrix[i, j] = corr_matrix[i, j] * std_devs[i] * std_devs[j]

    return cov_matrix


def main():
    df = pd.read_csv('problem4.csv')

    print("Part a")

    corr_matrix = exponential_weighted_correlation(df, lambda_param=0.94)

    print(f"{'':>8}", end="")
    for col in df.columns:
        print(f"{col:>15}", end="")
    print()

    for i, row_name in enumerate(df.columns):
        print(f"{row_name:>8}", end="")
        for j in range(len(df.columns)):
            print(f"{corr_matrix[i, j]:>15.10f}", end="")
        print()


    print("-" * 80)
    print("Part b")

    variances = exponential_weighted_variances(df, lambda_param=0.97)

    for i, col in enumerate(df.columns):
        print(f"Var({col}): {variances[i]:.15f}")

    print("-" * 80)
    print("Part c")

    combined_cov_matrix = combine_correlation_variance_to_covariance(corr_matrix, variances)

    print(f"{'':>8}", end="")
    for col in df.columns:
        print(f"{col:>15}", end="")
    print()

    for i, row_name in enumerate(df.columns):
        print(f"{row_name:>8}", end="")
        for j in range(len(df.columns)):
            print(f"{combined_cov_matrix[i, j]:>15.10f}", end="")
        print()


if __name__ == "__main__":
    main()
