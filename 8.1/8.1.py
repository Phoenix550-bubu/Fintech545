import pandas as pd
import numpy as np
from scipy import stats

def calculate_var_normal(data, alpha=0.05):
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    z_alpha = stats.norm.ppf(alpha)

    var_absolute = -(mean + std * z_alpha)

    var_diff_mean = -(std * z_alpha)

    return var_absolute, var_diff_mean

def main():
    data = pd.read_csv('test7_1.csv')
    x = data['x1'].values

    var_absolute, var_diff_mean = calculate_var_normal(x, alpha=0.05)

    result = pd.DataFrame({
        'VaR Absolute': [var_absolute],
        'VaR Diff from Mean': [var_diff_mean]
    })

    print(','.join(result.columns))

    for _, row in result.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()
