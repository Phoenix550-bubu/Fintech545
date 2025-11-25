import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq, minimize
import warnings
warnings.filterwarnings('ignore')


def calculate_arithmetic_returns(prices):
    returns = prices.pct_change().dropna()
    return returns


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


def gbsm(call, underlying, strike, ttm, rf, b, ivol):
    d1 = (np.log(underlying / strike) + (b + ivol**2 / 2) * ttm) / (ivol * np.sqrt(ttm))
    d2 = d1 - ivol * np.sqrt(ttm)

    if call:
        value = underlying * np.exp((b - rf) * ttm) * norm.cdf(d1) - strike * np.exp(-rf * ttm) * norm.cdf(d2)
        delta = np.exp((b - rf) * ttm) * norm.cdf(d1)
    else:
        value = strike * np.exp(-rf * ttm) * norm.cdf(-d2) - underlying * np.exp((b - rf) * ttm) * norm.cdf(-d1)
        delta = np.exp((b - rf) * ttm) * (norm.cdf(d1) - 1)

    return value, delta


def implied_volatility(option_price, call, underlying, strike, ttm, rf, b):
    def objective(ivol):
        value, _ = gbsm(call, underlying, strike, ttm, rf, b, ivol)
        return value - option_price

    try:
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
        return iv
    except:
        return np.nan


def simulate_returns(fitted_params, n_simulations, random_seed=42):
    dist_type, params = fitted_params
    np.random.seed(random_seed)

    if dist_type == 'Normal':
        mu, sigma = params
        simulated = stats.norm.rvs(loc=mu, scale=sigma, size=n_simulations)
    else:
        nu, mu, sigma = params
        simulated = stats.t.rvs(df=nu, loc=mu, scale=sigma, size=n_simulations)

    return simulated


def calculate_var_es(pnl, alpha=0.05):
    sorted_pnl = np.sort(pnl)
    var_index = int(np.floor(alpha * len(pnl)))
    var = -sorted_pnl[var_index]
    es_tail = sorted_pnl[:var_index]
    es = -np.mean(es_tail)
    return var, es


def calculate_portfolio_pnl(simulated_returns, current_price, call_strike, put_strike,
                           call_price, put_price, call_iv, put_iv, ttm_new, rf, div_rate,
                           stock_holding=1, call_holding=-1, put_holding=1):
    n_sim = len(simulated_returns)

    simulated_prices = current_price * (1 + simulated_returns)

    b = rf - div_rate
    call_values = np.zeros(n_sim)
    put_values = np.zeros(n_sim)

    for i in range(n_sim):
        call_values[i], _ = gbsm(True, simulated_prices[i], call_strike, ttm_new, rf, b, call_iv)
        put_values[i], _ = gbsm(False, simulated_prices[i], put_strike, ttm_new, rf, b, put_iv)

    stock_pnl = stock_holding * (simulated_prices - current_price)
    call_pnl = call_holding * (call_values - call_price)
    put_pnl = put_holding * (put_values - put_price)

    total_pnl = stock_pnl + call_pnl + put_pnl

    return total_pnl, simulated_prices, call_values, put_values


def main():
    rf = 0.04
    div_rate = 0.0109
    days_to_expiration = 10
    trading_days_per_year = 255
    ttm = days_to_expiration / trading_days_per_year

    call_strike = 665
    call_price = 7.05
    put_strike = 655
    put_price = 7.69

    holding_days = 5
    ttm_new = (days_to_expiration - holding_days) / trading_days_per_year
    rf_holding = rf * holding_days / trading_days_per_year

    df = pd.read_csv('problem2.csv')
    prices = df['SPY'].values
    current_price = prices[-1]

    returns = calculate_arithmetic_returns(df['SPY'])
    returns_data = returns.values

    print("=" * 80)
    print("Part a: Best Fit Model")

    mu_norm, sigma_norm, ll_norm, aic_norm, bic_norm = fit_normal_distribution(returns_data)
    print("\nNormal Distribution:")
    print(f"  μ:              {mu_norm:.10f}")
    print(f"  σ:              {sigma_norm:.10f}")
    print(f"  Log-Likelihood: {ll_norm:.6f}")
    print(f"  AIC:            {aic_norm:.6f}")
    print(f"  BIC:            {bic_norm:.6f}")

    nu_t, mu_t, sigma_t, ll_t, aic_t, bic_t = fit_t_distribution(returns_data)
    print("\nT-Distribution:")
    print(f"  ν:              {nu_t:.10f}")
    print(f"  μ:              {mu_t:.10f}")
    print(f"  σ:              {sigma_t:.10f}")
    print(f"  Log-Likelihood: {ll_t:.6f}")
    print(f"  AIC:            {aic_t:.6f}")
    print(f"  BIC:            {bic_t:.6f}")

    if aic_t < aic_norm:
        best_model = ('T', (nu_t, mu_t, sigma_t))
    else:
        best_model = ('Normal', (mu_norm, sigma_norm))

    print("\n" + "=" * 80)
    print("Part b: Implied Volatility")

    b = rf - div_rate

    call_iv = implied_volatility(call_price, True, current_price, call_strike, ttm, rf, b)
    put_iv = implied_volatility(put_price, False, current_price, put_strike, ttm, rf, b)

    print(f"\nCurrent SPY Price: ${current_price:.2f}")
    print(f"\nCall Option (Strike=${call_strike}, Price=${call_price}):")
    print(f"  Implied Volatility: {call_iv:.10f}")
    print(f"\nPut Option (Strike=${put_strike}, Price=${put_price}):")
    print(f"  Implied Volatility: {put_iv:.10f}")

    print("\n" + "=" * 80)
    print("Part c: Portfolio VaR and ES")

    current_portfolio_value = current_price - call_price + put_price
    print(f"\nCurrent Portfolio Value: ${current_portfolio_value:.2f}")
    print(f"  Long 1 Stock:  ${current_price:.2f}")
    print(f"  Short 1 Call: -${call_price:.2f}")
    print(f"  Long 1 Put:   +${put_price:.2f}")

    n_simulations = 50000
    simulated_returns_1day = simulate_returns(best_model, n_simulations, random_seed=42)

    simulated_returns_5day = simulated_returns_1day * np.sqrt(holding_days)

    portfolio_pnl, _, _, _ = calculate_portfolio_pnl(
        simulated_returns_5day, current_price, call_strike, put_strike,
        call_price, put_price, call_iv, put_iv, ttm_new, rf, div_rate,
        stock_holding=1, call_holding=-1, put_holding=1
    )

    var_dollar, es_dollar = calculate_var_es(portfolio_pnl, alpha=0.05)
    var_pct = var_dollar / current_portfolio_value
    es_pct = es_dollar / current_portfolio_value

    print(f"\nVaR (5%, absolute): ${var_dollar:.2f}")
    print(f"VaR (5%, %):        {var_pct*100:.4f}%")
    print(f"ES (5%, absolute):  ${es_dollar:.2f}")
    print(f"ES (5%, %):         {es_pct*100:.4f}%")

    mean_pnl = np.mean(portfolio_pnl)
    ratio_current = (mean_pnl - rf_holding) / es_dollar
    print(f"\nCurrent Portfolio Ratio: (mean - rf) / ES = {ratio_current:.4f}")


if __name__ == "__main__":
    main()
