import pandas as pd
import numpy as np
from scipy.optimize import minimize


def portfolio_volatility(weights, covar):
    return np.sqrt(weights @ covar @ weights)


def component_sd(weights, covar):
    pVol = portfolio_volatility(weights, covar)
    csd = weights * (covar @ weights) / pVol
    return csd


def sse_csd(weights, covar):
    csd = component_sd(weights, covar)
    mean_csd = np.mean(csd)
    diff = csd - mean_csd
    sse = np.sum(diff ** 2)
    return sse * 1e5


def risk_parity_optimization(covar, initial_weights=None):
    n = covar.shape[0]

    if initial_weights is None:
        initial_weights = np.ones(n) / n

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    bounds = tuple((0, None) for _ in range(n))

    result = minimize(
        sse_csd,
        initial_weights,
        args=(covar,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    return result.x


def main():
    covar_df = pd.read_csv('test5_2.csv')
    covar = covar_df.values

    weights = risk_parity_optimization(covar)

    weights = weights / np.sum(weights)

    print('W')
    for w in weights:
        print(w)


if __name__ == "__main__":
    main()
