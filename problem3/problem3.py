import pandas as pd
import numpy as np
from scipy import stats
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')


def calculate_var_normal(data, alpha=0.05):
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    z_alpha = stats.norm.ppf(alpha)

    var_from_zero = -(mean + std * z_alpha)

    var_from_mean = -(std * z_alpha)

    return var_from_zero, var_from_mean


def calculate_var_t(data, alpha=0.05):
    params = stats.t.fit(data)
    nu = params[0]
    mu = params[1]
    sigma = params[2]

    t_alpha = stats.t.ppf(alpha, nu)

    var_from_zero = -(mu + sigma * t_alpha)

    var_from_mean = -(sigma * t_alpha)

    return var_from_zero, var_from_mean, nu, mu, sigma


def calculate_es_normal(data, alpha=0.05):
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    z_alpha = stats.norm.ppf(alpha)

    phi_z_alpha = stats.norm.pdf(z_alpha)

    es_from_mean = std * phi_z_alpha / alpha

    es_from_zero = -(mean - std * phi_z_alpha / alpha)

    return es_from_zero, es_from_mean


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

    es_from_zero = -conditional_expectation

    es_from_mean = mu - conditional_expectation

    return es_from_zero, es_from_mean, nu, mu, sigma


def main():
    df = pd.read_csv('problem2.csv')
    data = df['X'].values

    print("Part a")

    var_norm_zero, var_norm_mean = calculate_var_normal(data, alpha=0.05)
    print("Normal Distribution VaR:")
    print(f"  VaR from 0:    {var_norm_zero:.10f}")
    print(f"  VaR from mean: {var_norm_mean:.10f}")

    var_t_zero, var_t_mean, nu_t, mu_t, sigma_t = calculate_var_t(data, alpha=0.05)
    print("T-Distribution VaR:")
    print(f"  ν:             {nu_t:.6f}")
    print(f"  μ:             {mu_t:.10f}")
    print(f"  σ:             {sigma_t:.10f}")
    print(f"  VaR from 0:    {var_t_zero:.10f}")
    print(f"  VaR from mean: {var_t_mean:.10f}")


    print("-" * 80)
    print("Part b")

    es_norm_zero, es_norm_mean = calculate_es_normal(data, alpha=0.05)
    print("Normal Distribution ES:")
    print(f"  ES from 0:    {es_norm_zero:.10f}")
    print(f"  ES from mean: {es_norm_mean:.10f}")

    es_t_zero, es_t_mean, nu_t_es, mu_t_es, sigma_t_es = calculate_es_t(data, alpha=0.05)
    print("T-Distribution ES:")
    print(f"  ν:            {nu_t_es:.6f}")
    print(f"  μ:            {mu_t_es:.10f}")
    print(f"  σ:            {sigma_t_es:.10f}")
    print(f"  ES from 0:    {es_t_zero:.10f}")
    print(f"  ES from mean: {es_t_mean:.10f}")


    print("-" * 80)
    print("Part c")
    print(f"ES/VaR Ratio: {es_norm_zero / var_norm_zero:.4f}")
    print(f"ES/VaR Ratio: {es_t_zero / var_t_zero:.4f}")

    var_diff_pct = ((var_t_zero / var_norm_zero - 1) * 100)
    es_diff_pct = ((es_t_zero / es_norm_zero - 1) * 100)

    print(f"{var_diff_pct:.2f}%")
    print(f"{es_diff_pct:.2f}%")

if __name__ == "__main__":
    main()
