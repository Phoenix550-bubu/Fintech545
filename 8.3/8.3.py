import pandas as pd
import numpy as np
from scipy import stats

def calculate_var_simulation(data, alpha=0.05, n_simulations=100000, random_seed=42):
    params = stats.t.fit(data)
    nu = params[0]
    mu = params[1]
    sigma = params[2]

    np.random.seed(random_seed)

    simulated_returns = stats.t.rvs(nu, loc=mu, scale=sigma, size=n_simulations)

    simulated_sorted = np.sort(simulated_returns)

    var_index = int(np.floor(alpha * n_simulations))

    percentile_alpha = simulated_sorted[var_index]

    var_absolute = -percentile_alpha

    var_diff_mean = mu - percentile_alpha

    return var_absolute, var_diff_mean

def main():
    data = pd.read_csv('test7_2.csv')
    x = data['x1'].values

    var_absolute, var_diff_mean = calculate_var_simulation(
        x,
        alpha=0.05,
        n_simulations=100000,
        random_seed=42
    )

    result = pd.DataFrame({
        'VaR Absolute': [var_absolute],
        'VaR Diff from Mean': [var_diff_mean]
    })

    print(','.join(result.columns))
    for _, row in result.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()
