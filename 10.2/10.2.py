import pandas as pd
import numpy as np
from scipy.optimize import minimize


def portfolio_volatility(weights, covar):
    return np.sqrt(weights @ covar @ weights)


def component_sd(weights, covar):
    pVol = portfolio_volatility(weights, covar)
    csd = weights * (covar @ weights) / pVol
    return csd


def sse_csd_with_risk_budget(weights, covar, risk_budget):
    csd = component_sd(weights, covar)

    adjusted_csd = csd / risk_budget

    mean_adjusted_csd = np.mean(adjusted_csd)

    diff = adjusted_csd - mean_adjusted_csd
    sse = np.sum(diff ** 2)

    return sse * 1e5


def risk_parity_with_budget(covar, risk_budget, initial_weights=None):
    n = covar.shape[0]

    if initial_weights is None:
        initial_weights = np.ones(n) / n

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    bounds = tuple((0, None) for _ in range(n))

    result = minimize(
        sse_csd_with_risk_budget,
        initial_weights,
        args=(covar, risk_budget),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )

    return result.x


def main():
    covar_df = pd.read_csv('test5_2.csv')
    covar = covar_df.values

    risk_budget = np.array([1.4044417, 1.4044417, 1.0, 1.0, 0.5])

    weights = risk_parity_with_budget(covar, risk_budget)

    weights = weights / np.sum(weights)

    print('W')
    for w in weights:
        print(w)


if __name__ == "__main__":
    main()
