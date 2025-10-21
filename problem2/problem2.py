import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def calculate_moments(data):
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    skewness = stats.skew(data, bias=False)
    kurtosis = stats.kurtosis(data, bias=False, fisher=True)
    return mean, variance, skewness, kurtosis


def fit_normal_distribution(data):
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)

    log_likelihood = np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

    n_params = 2
    n_obs = len(data)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood
    
    return mu, sigma, log_likelihood, aic, bic


def fit_t_distribution(data):
    params = stats.t.fit(data)
    nu = params[0]
    mu = params[1]
    sigma = params[2]

    log_likelihood = np.sum(stats.t.logpdf(data, df=nu, loc=mu, scale=sigma))

    n_params = 3
    n_obs = len(data)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood
    
    return nu, mu, sigma, log_likelihood, aic, bic


def compare_distributions(data, normal_params, t_params):
    actual_mean, actual_var, actual_skew, actual_kurt = calculate_moments(data)

    mu_norm, sigma_norm = normal_params[:2]
    normal_mean = mu_norm
    normal_var = sigma_norm ** 2
    normal_skew = 0
    normal_kurt = 0

    nu_t, mu_t, sigma_t = t_params[:3]
    t_mean = mu_t if nu_t > 1 else np.nan
    t_var = sigma_t ** 2 * nu_t / (nu_t - 2) if nu_t > 2 else np.nan
    t_skew = 0 if nu_t > 3 else np.nan
    t_kurt = 6 / (nu_t - 4) if nu_t > 4 else np.nan
    
    return {
        'actual': (actual_mean, actual_var, actual_skew, actual_kurt),
        'normal': (normal_mean, normal_var, normal_skew, normal_kurt),
        't': (t_mean, t_var, t_skew, t_kurt)
    }


def perform_goodness_of_fit_tests(data, normal_params, t_params):
    mu_norm, sigma_norm = normal_params[:2]
    nu_t, mu_t, sigma_t = t_params[:3]

    ks_stat_norm, ks_pval_norm = stats.kstest(data, 'norm', args=(mu_norm, sigma_norm))

    ks_stat_t, ks_pval_t = stats.kstest(data, 't', args=(nu_t, mu_t, sigma_t))
    
    return {
        'normal': (ks_stat_norm, ks_pval_norm),
        't': (ks_stat_t, ks_pval_t)
    }


def main():
    df = pd.read_csv('problem2.csv')
    data = df['X'].values

    print("Part a")
    mean, variance, skewness, kurtosis = calculate_moments(data)

    print(f"Mean:     {mean:.10f}")
    print(f"Variance: {variance:.10f}")
    print(f"Skewness: {skewness:.10f}")
    print(f"Kurtosis: {kurtosis:.10f}")


    print("-" * 80)
    print("Part c")
    mu_norm, sigma_norm, ll_norm, aic_norm, bic_norm = fit_normal_distribution(data)
    print("Normal Distribution:")
    print(f"  μ:              {mu_norm:.10f}")
    print(f"  σ:              {sigma_norm:.10f}")
    print(f"  Log-Likelihood: {ll_norm:.6f}")
    print(f"  AIC:            {aic_norm:.6f}")
    print(f"  BIC:            {bic_norm:.6f}")

    nu_t, mu_t, sigma_t, ll_t, aic_t, bic_t = fit_t_distribution(data)
    print("T-Distribution:")
    print(f"  ν:              {nu_t:.10f}")
    print(f"  μ:              {mu_t:.10f}")
    print(f"  σ:              {sigma_t:.10f}")
    print(f"  Log-Likelihood: {ll_t:.6f}")
    print(f"  AIC:            {aic_t:.6f}")
    print(f"  BIC:            {bic_t:.6f}")

if __name__ == "__main__":
    main()

