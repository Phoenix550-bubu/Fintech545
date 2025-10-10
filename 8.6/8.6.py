import pandas as pd
import numpy as np
from scipy import stats

def calculate_es_simulation(data, alpha=0.05, n_simulations=100000, random_seed=42):
    params = stats.t.fit(data)
    nu = params[0]
    mu = params[1]
    sigma = params[2]

    np.random.seed(random_seed)

    simulated_returns = stats.t.rvs(nu, loc=mu, scale=sigma, size=n_simulations)

    simulated_sorted = np.sort(simulated_returns)

    var_index = int(np.floor(alpha * n_simulations))

    es_tail = simulated_sorted[:var_index]
    es_mean = np.mean(es_tail)

    es_absolute = -es_mean

    es_diff_mean = mu - es_mean

    return es_absolute, es_diff_mean

def main():
    data = pd.read_csv('test7_2.csv')
    x = data['x1'].values

    es_absolute, es_diff_mean = calculate_es_simulation(
        x,
        alpha=0.05,
        n_simulations=100000,
        random_seed=42
    )

    result = pd.DataFrame({
        'ES Absolute': [es_absolute],
        'ES Diff from Mean': [es_diff_mean]
    })

    print(','.join(result.columns))

    for _, row in result.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()
