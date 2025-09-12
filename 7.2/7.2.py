import pandas as pd
import numpy as np
from scipy import stats

def fit_t_distribution(data):
    df = pd.read_csv(data)
    data = df['x1'].dropna().values

    params = stats.t.fit(data)
    nu = params[0]
    mu = params[1]
    sigma = params[2]

    print(f"mu: {mu:.18f}")
    print(f"sigma: {sigma:.18f}")
    print(f"nu: {nu:.18f}")

    return mu, sigma, nu

if __name__ == "__main__":
    mu, sigma, nu = fit_t_distribution('test7_2.csv')