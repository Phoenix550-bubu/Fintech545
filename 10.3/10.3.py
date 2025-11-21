import pandas as pd
import numpy as np
from scipy.optimize import minimize


def sharpe_ratio(weights, mean_returns, covar, rf):
    portfolio_return = weights @ mean_returns
    portfolio_std = np.sqrt(weights @ covar @ weights)
    sharpe = (portfolio_return - rf) / portfolio_std
    return sharpe


def negative_sharpe_ratio(weights, mean_returns, covar, rf):
    return -sharpe_ratio(weights, mean_returns, covar, rf)


def max_sharpe_ratio_optimization(mean_returns, covar, rf):
    n = len(mean_returns)

    initial_weights = np.ones(n) / n

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    bounds = tuple((0, None) for _ in range(n))

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, covar, rf),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )

    return result.x


def main():
    covar_df = pd.read_csv('test5_2.csv')
    covar = covar_df.values

    means_df = pd.read_csv('test10_3_means.csv')
    mean_returns = means_df['Mean'].values

    rf = 0.04

    weights = max_sharpe_ratio_optimization(mean_returns, covar, rf)

    weights = weights / np.sum(weights)

    print('W')
    for w in weights:
        print(w)


if __name__ == "__main__":
    main()
