import pandas as pd
import numpy as np
from scipy import stats

def fit_normal_distribution(data):
    df = pd.read_csv(data)
    data = df['x1'].dropna().values

    mu = np.mean(data)
    sigma = np.std(data, ddof=1)

    print(f"mu: {mu:.18f}")
    print(f"sigma: {sigma:.18f}")

    return mu, sigma

if __name__ == "__main__":
    mu, sigma = fit_normal_distribution('test7_1.csv')