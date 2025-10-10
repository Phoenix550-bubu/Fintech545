import pandas as pd
import numpy as np
from scipy import stats

def calculate_es_normal(data, alpha=0.05):
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    z_alpha = stats.norm.ppf(alpha)

    phi_z_alpha = stats.norm.pdf(z_alpha)

    es_diff_mean = std * phi_z_alpha / alpha

    es_absolute = -(mean - std * phi_z_alpha / alpha)

    return es_absolute, es_diff_mean

def main():
    data = pd.read_csv('test7_1.csv')
    x = data['x1'].values

    es_absolute, es_diff_mean = calculate_es_normal(x, alpha=0.05)

    result = pd.DataFrame({
        'ES Absolute': [es_absolute],
        'ES Diff from Mean': [es_diff_mean]
    })

    print(','.join(result.columns))

    for _, row in result.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()
