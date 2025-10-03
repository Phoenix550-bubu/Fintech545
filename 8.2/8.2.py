import pandas as pd
import numpy as np
from scipy import stats

def calculate_var_t(data, alpha=0.05):
    params = stats.t.fit(data)
    nu = params[0]
    mu = params[1]
    sigma = params[2]

    t_alpha = stats.t.ppf(alpha, nu)

    var_absolute = -(mu + sigma * t_alpha)

    var_diff_mean = -(sigma * t_alpha)

    return var_absolute, var_diff_mean

def main():
    data = pd.read_csv('test7_2.csv')
    x = data['x1'].values

    var_absolute, var_diff_mean = calculate_var_t(x, alpha=0.05)

    result = pd.DataFrame({
        'VaR Absolute': [var_absolute],
        'VaR Diff from Mean': [var_diff_mean]
    })

    print(','.join(result.columns))

    for _, row in result.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()
