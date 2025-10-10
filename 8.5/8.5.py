import pandas as pd
import numpy as np
from scipy import stats
from scipy.integrate import quad

def calculate_es_t(data, alpha=0.05):
    params = stats.t.fit(data)
    nu = params[0]
    mu = params[1]
    sigma = params[2]

    t_alpha = stats.t.ppf(alpha, nu)

    var_percentile = mu + sigma * t_alpha

    def integrand(x):
        return x * stats.t.pdf((x - mu) / sigma, nu) / sigma

    lower_bound = mu + sigma * stats.t.ppf(1e-12, nu)

    integral_result, error = quad(integrand, lower_bound, var_percentile)

    conditional_expectation = integral_result / alpha

    es_absolute = -conditional_expectation

    es_diff_mean = mu - conditional_expectation

    return es_absolute, es_diff_mean

def main():
    data = pd.read_csv('test7_2.csv')
    x = data['x1'].values

    es_absolute, es_diff_mean = calculate_es_t(x, alpha=0.05)

    result = pd.DataFrame({
        'ES Absolute': [es_absolute],
        'ES Diff from Mean': [es_diff_mean]
    })

    print(','.join(result.columns))

    for _, row in result.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()
