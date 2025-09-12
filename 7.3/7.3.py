import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def fit_t_regression(data):
    df = pd.read_csv(data)
    X = df[['x1', 'x2', 'x3']].values
    y = df['y'].values
    n = len(y)
    X_design = np.column_stack([np.ones(n), X])

    # Initialize parameters using OLS estimates
    beta_ols = np.linalg.lstsq(X_design, y, rcond=None)[0]
    residuals_ols = y - X_design @ beta_ols
    sigma_init = np.std(residuals_ols, ddof=len(beta_ols))

    # Better nu initialization based on residual kurtosis
    residual_kurtosis = stats.kurtosis(residuals_ols, fisher=True)
    if residual_kurtosis > 0:
        nu_init = max(3.0, 6.0 / residual_kurtosis + 4.0)
    else:
        nu_init = 8.0

    # Parameter order: [nu, alpha, b1, b2, b3, sigma]
    initial_params = np.array([nu_init, beta_ols[0], beta_ols[1], beta_ols[2], beta_ols[3], sigma_init])

    # Define negative log-likelihood function
    def neg_log_likelihood(params):
        nu, alpha, b1, b2, b3, sigma = params

        if nu <= 2.0 or sigma <= 1e-8:
            return np.inf

        beta = np.array([alpha, b1, b2, b3])
        residuals = y - X_design @ beta

        # T-distribution log-likelihood with numerical stability
        standardized_residuals = residuals / sigma
        standardized_residuals = np.clip(standardized_residuals, -50, 50)
        log_likelihood = np.sum(stats.t.logpdf(standardized_residuals, df=nu)) - n * np.log(sigma)

        return -log_likelihood

    # Optimize parameters with multiple starting points
    bounds = [(2.01, 50), (-np.inf, np.inf), (-np.inf, np.inf),
              (-np.inf, np.inf), (-np.inf, np.inf), (1e-8, 1.0)]

    best_result = None
    best_likelihood = np.inf

    nu_starts = [3.0, 4.0, 5.0, 6.0, nu_init]
    for nu_start in nu_starts:
        start_params = initial_params.copy()
        start_params[0] = nu_start

        result = minimize(neg_log_likelihood, start_params, method='L-BFGS-B', bounds=bounds,
                         options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 1000})

        if result.success and result.fun < best_likelihood:
            best_result = result
            best_likelihood = result.fun

    if best_result is None:
        result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B', bounds=bounds)
        best_result = result

    mu = 0.0
    nu_opt, alpha_opt, b1_opt, b2_opt, b3_opt, sigma_opt = best_result.x

    print(f"mu: {mu}")
    print(f"sigma: {sigma_opt:.18f}")
    print(f"nu: {nu_opt:.18f}")
    print(f"Alpha: {alpha_opt:.18f}")
    print(f"B1: {b1_opt:.18f}")
    print(f"B2: {b2_opt:.18f}")
    print(f"B3: {b3_opt:.18f}")

    return mu, sigma_opt, nu_opt, alpha_opt, b1_opt, b2_opt, b3_opt

if __name__ == "__main__":
    mu, sigma, nu, Alpha, B1, B2, B3 = fit_t_regression('test7_3.csv')
