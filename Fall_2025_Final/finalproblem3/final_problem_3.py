import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def exponential_weights(n, lambda_param):
    weights = np.zeros(n)
    total_weight = 0.0

    for i in range(n):
        weights[i] = (1 - lambda_param) * (lambda_param ** (i + 1))
        total_weight += weights[i]

    weights = weights / total_weight
    return weights


def exponential_weighted_covariance(data, lambda_param=0.97):
    X = data.values
    n, m = X.shape

    weights = exponential_weights(n, lambda_param)
    weights = weights[::-1]

    weighted_means = np.zeros(m)
    for j in range(m):
        weighted_means[j] = np.sum(weights * X[:, j])

    cov_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            weighted_cov = 0.0
            for t in range(n):
                weighted_cov += weights[t] * (X[t, i] - weighted_means[i]) * (X[t, j] - weighted_means[j])
            cov_matrix[i, j] = weighted_cov

    return cov_matrix


def portfolio_volatility(weights, covar):
    return np.sqrt(weights @ covar @ weights)


def component_sd(weights, covar):
    pVol = portfolio_volatility(weights, covar)
    csd = weights * (covar @ weights) / pVol
    return csd


def sse_csd(weights, covar):
    csd = component_sd(weights, covar)
    mean_csd = np.mean(csd)
    diff = csd - mean_csd
    sse = np.sum(diff ** 2)
    return sse * 1e5


def risk_parity_optimization(covar, initial_weights=None):
    n = covar.shape[0]

    if initial_weights is None:
        initial_weights = np.ones(n) / n

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = tuple((0, None) for _ in range(n))

    result = minimize(
        sse_csd,
        initial_weights,
        args=(covar,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    return result.x


def sharpe_ratio(weights, mean_returns, covar, rf):
    portfolio_return = weights @ mean_returns
    portfolio_std = np.sqrt(weights @ covar @ weights)
    sharpe = (portfolio_return - rf) / portfolio_std
    return sharpe


def negative_sharpe_ratio(weights, mean_returns, covar, rf):
    return -sharpe_ratio(weights, mean_returns, covar, rf)


def max_sharpe_ratio_optimization(mean_returns, covar, rf):
    n = len(mean_returns)

    initial_weights = np.ones(n) / n
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = tuple((0, None) for _ in range(n))

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, covar, rf),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )

    return result.x


def expost_attribution(returns, initial_weights):
    asset_names = returns.columns.tolist()
    n_assets = len(asset_names)

    returns_matrix = returns.values
    n_periods = len(returns_matrix)

    portfolio_returns = np.zeros(n_periods)
    weights_matrix = np.zeros((n_periods, n_assets))
    current_weights = initial_weights.copy()

    for i in range(n_periods):
        weights_matrix[i, :] = current_weights
        asset_returns = returns_matrix[i, :]
        current_weights = current_weights * (1.0 + asset_returns)
        pR = np.sum(current_weights)
        current_weights = current_weights / pR
        portfolio_returns[i] = pR - 1

    portfolio_total_return = np.exp(np.sum(np.log(1.0 + portfolio_returns))) - 1
    K = np.log(1 + portfolio_total_return) / portfolio_total_return
    carino_k = np.log(1.0 + portfolio_returns) / portfolio_returns / K

    return_attribution = {}
    for i, asset in enumerate(asset_names):
        asset_returns = returns_matrix[:, i]
        weights_over_time = weights_matrix[:, i]
        attr = np.sum(asset_returns * weights_over_time * carino_k)
        return_attribution[asset] = attr

    Y = returns_matrix * weights_matrix
    X = np.column_stack([np.ones(len(portfolio_returns)), portfolio_returns])
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    beta = B[1, :]

    portfolio_std = np.std(portfolio_returns, ddof=1)
    component_sd_values = beta * portfolio_std

    vol_attribution = {}
    for i, asset in enumerate(asset_names):
        vol_attribution[asset] = component_sd_values[i]

    return return_attribution, vol_attribution, portfolio_total_return, portfolio_std


def main():
    insample_df = pd.read_csv('problem3_insample.csv')
    outsample_df = pd.read_csv('problem3_outsample.csv')

    insample_returns = insample_df.drop('Date', axis=1)
    outsample_returns = outsample_df.drop('Date', axis=1)

    stocks = insample_returns.columns.tolist()
    rf = 0.04

    print("Part a: Portfolio Construction using In-Sample Data")

    insample_mean_monthly = insample_returns.mean().values

    insample_mean_annual = (1 + insample_mean_monthly) ** 12 - 1

    ewm_cov_monthly = exponential_weighted_covariance(insample_returns, lambda_param=0.97)

    ewm_cov_annual = ewm_cov_monthly * 12

    print("\nIn-Sample Expected Returns:")
    for i, stock in enumerate(stocks):
        print(f"  {stock}: {insample_mean_annual[i]:.10f}")

    print("\nExponentially Weighted Covariance Matrix:")
    print(f"{'':>8}", end="")
    for stock in stocks:
        print(f"{stock:>15}", end="")
    print()
    for i, stock in enumerate(stocks):
        print(f"{stock:>8}", end="")
        for j in range(len(stocks)):
            print(f"{ewm_cov_annual[i, j]:>15.10f}", end="")
        print()

    print("\n" + "-" * 80)
    print("Max Sharpe Ratio Portfolio:")

    msr_weights = max_sharpe_ratio_optimization(insample_mean_annual, ewm_cov_annual, rf)
    msr_weights = msr_weights / np.sum(msr_weights)

    msr_return = msr_weights @ insample_mean_annual
    msr_vol = portfolio_volatility(msr_weights, ewm_cov_annual)
    msr_sharpe = (msr_return - rf) / msr_vol

    print("\nWeights:")
    for i, stock in enumerate(stocks):
        print(f"  {stock}: {msr_weights[i]:.10f}")

    print(f"\nExpected Return: {msr_return:.10f}")
    print(f"Volatility:      {msr_vol:.10f}")
    print(f"Sharpe Ratio:    {msr_sharpe:.10f}")

    msr_csd = component_sd(msr_weights, ewm_cov_annual)

    print("\n" + "-" * 80)
    print("Risk Parity Portfolio:")

    rp_weights = risk_parity_optimization(ewm_cov_annual)
    rp_weights = rp_weights / np.sum(rp_weights)

    rp_return = rp_weights @ insample_mean_annual
    rp_vol = portfolio_volatility(rp_weights, ewm_cov_annual)
    rp_sharpe = (rp_return - rf) / rp_vol

    print("\nWeights:")
    for i, stock in enumerate(stocks):
        print(f"  {stock}: {rp_weights[i]:.10f}")

    print(f"\nExpected Return: {rp_return:.10f}")
    print(f"Volatility:      {rp_vol:.10f}")
    print(f"Sharpe Ratio:    {rp_sharpe:.10f}")

    rp_csd = component_sd(rp_weights, ewm_cov_annual)

    print("\n" + "=" * 80)
    print("Part b: Out-of-Sample Performance and Attribution")

    print("Max Sharpe Ratio Portfolio Out of Sample:")

    msr_return_attr, msr_vol_attr, msr_total_return, msr_portfolio_std = expost_attribution(
        outsample_returns, msr_weights
    )

    print("\nEx-Post Return Attribution:")
    total_attr = 0
    for stock in stocks:
        print(f"  {stock}: {msr_return_attr[stock]:.10f}")
        total_attr += msr_return_attr[stock]

    msr_portfolio_std_annual = msr_portfolio_std * np.sqrt(12)

    print(f"\nEx-Post Volatility (Monthly):  {msr_portfolio_std:.10f}")
    print(f"Ex-Post Volatility (Annual):   {msr_portfolio_std_annual:.10f}")

    print("\nEx-Post Risk Attribution:")
    total_vol_attr = 0
    for stock in stocks:
        msr_vol_attr_annual = msr_vol_attr[stock] * np.sqrt(12)
        print(f"  {stock}: {msr_vol_attr_annual:.10f}")
        total_vol_attr += msr_vol_attr_annual

    print("\n  Ex-Ante vs Ex-Post Risk Attribution:")
    print(f"  {'Stock':>8} {'Ex-Ante (Abs)':>18} {'Ex-Ante (%)':>15} {'Ex-Post (Abs)':>18} {'Ex-Post (%)':>15}")
    for i, stock in enumerate(stocks):
        msr_vol_attr_annual = msr_vol_attr[stock] * np.sqrt(12)
        print(f"  {stock:>8} {msr_csd[i]:>18.10f} {msr_csd[i]/msr_vol*100:>14.4f}% "
              f"{msr_vol_attr_annual:>18.10f} {msr_vol_attr_annual/msr_portfolio_std_annual*100:>14.4f}%")

    print("\n" + "-" * 80)
    print("Risk Parity Portfolio - Out-of-Sample:")

    rp_return_attr, rp_vol_attr, rp_total_return, rp_portfolio_std = expost_attribution(
        outsample_returns, rp_weights
    )

    print("\nEx-Post Return Attribution:")
    total_attr = 0
    for stock in stocks:
        print(f"  {stock}: {rp_return_attr[stock]:.10f}")
        total_attr += rp_return_attr[stock]

    rp_portfolio_std_annual = rp_portfolio_std * np.sqrt(12)

    print(f"\nEx-Post Volatility (Monthly):  {rp_portfolio_std:.10f}")
    print(f"Ex-Post Volatility (Annual):   {rp_portfolio_std_annual:.10f}")

    print("\nEx-Post Risk Attribution:")
    total_vol_attr = 0
    for stock in stocks:
        rp_vol_attr_annual = rp_vol_attr[stock] * np.sqrt(12)
        print(f"  {stock}: {rp_vol_attr_annual:.10f}")
        total_vol_attr += rp_vol_attr_annual

    print("\n  Ex-Ante vs Ex-Post Risk Attribution:")
    print(f"  {'Stock':>8} {'Ex-Ante (Abs)':>18} {'Ex-Ante (%)':>15} {'Ex-Post (Abs)':>18} {'Ex-Post (%)':>15}")
    for i, stock in enumerate(stocks):
        rp_vol_attr_annual = rp_vol_attr[stock] * np.sqrt(12)
        print(f"  {stock:>8} {rp_csd[i]:>18.10f} {rp_csd[i]/rp_vol*100:>14.4f}% "
              f"{rp_vol_attr_annual:>18.10f} {rp_vol_attr_annual/rp_portfolio_std_annual*100:>14.4f}%")


if __name__ == "__main__":
    main()
