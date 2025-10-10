import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr

def fit_distribution(data, dist_type):
    if dist_type == 'Normal':
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)
        return stats.norm(loc=mu, scale=sigma)
    elif dist_type == 'T':
        params = stats.t.fit(data)
        nu = params[0]
        mu = params[1]
        sigma = params[2]
        return stats.t(df=nu, loc=mu, scale=sigma)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

def calculate_var_es_copula(returns_df, portfolio_df, alpha=0.05, n_simulations=100000, random_seed=42):
    np.random.seed(random_seed)

    stocks = portfolio_df['Stock'].values
    n_stocks = len(stocks)

    # Fit marginal distributions
    fitted_dists = {}
    for stock in stocks:
        returns = returns_df[stock].values
        dist_type = portfolio_df[portfolio_df['Stock'] == stock]['Distribution'].values[0]
        fitted_dists[stock] = fit_distribution(returns, dist_type)

    # Transform returns to uniform [0,1] using CDF
    U = np.zeros((len(returns_df), n_stocks))
    for i, stock in enumerate(stocks):
        returns = returns_df[stock].values
        U[:, i] = fitted_dists[stock].cdf(returns)

    # Transform uniform to standard normal
    Z = stats.norm.ppf(U)

    # Calculate Spearman correlation of Z
    spearman_corr = np.eye(n_stocks)
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            corr, _ = spearmanr(Z[:, i], Z[:, j])
            spearman_corr[i, j] = corr
            spearman_corr[j, i] = corr

    # Simulate from multivariate normal with Spearman correlation
    mean_vec = np.zeros(n_stocks)
    simulated_Z = np.random.multivariate_normal(mean_vec, spearman_corr, size=n_simulations)

    # Transform simulated Z to uniform using normal CDF
    simulated_U = stats.norm.cdf(simulated_Z)

    # Transform uniform to returns using inverse CDF
    simulated_returns = np.zeros((n_simulations, n_stocks))
    for i, stock in enumerate(stocks):
        simulated_returns[:, i] = fitted_dists[stock].ppf(simulated_U[:, i])

    # Calculate portfolio values and VaR/ES
    results = []

    for i, stock in enumerate(stocks):
        holding = portfolio_df[portfolio_df['Stock'] == stock]['Holding'].values[0]
        starting_price = portfolio_df[portfolio_df['Stock'] == stock]['Starting Price'].values[0]

        current_value = holding * starting_price

        simulated_prices = starting_price * (1 + simulated_returns[:, i])
        simulated_values = holding * simulated_prices

        simulated_pnl = simulated_values - current_value

        sorted_pnl = np.sort(simulated_pnl)

        var_index = int(np.floor(alpha * n_simulations))
        var_absolute = -sorted_pnl[var_index]
        es_tail = sorted_pnl[:var_index]
        es_absolute = -np.mean(es_tail)

        var_pct = var_absolute / current_value
        es_pct = es_absolute / current_value

        results.append({
            'Stock': stock,
            'VaR95': var_absolute,
            'ES95': es_absolute,
            'VaR95_Pct': var_pct,
            'ES95_Pct': es_pct
        })

    # Total portfolio value and simulated values
    total_current_value = 0
    total_simulated_values = np.zeros(n_simulations)

    for i, stock in enumerate(stocks):
        holding = portfolio_df[portfolio_df['Stock'] == stock]['Holding'].values[0]
        starting_price = portfolio_df[portfolio_df['Stock'] == stock]['Starting Price'].values[0]

        current_value = holding * starting_price
        total_current_value += current_value

        simulated_prices = starting_price * (1 + simulated_returns[:, i])
        simulated_values = holding * simulated_prices
        total_simulated_values += simulated_values

    total_pnl = total_simulated_values - total_current_value
    sorted_total_pnl = np.sort(total_pnl)

    var_index = int(np.floor(alpha * n_simulations))
    total_var_absolute = -sorted_total_pnl[var_index]
    total_es_tail = sorted_total_pnl[:var_index]
    total_es_absolute = -np.mean(total_es_tail)

    total_var_pct = total_var_absolute / total_current_value
    total_es_pct = total_es_absolute / total_current_value

    results.append({
        'Stock': 'Total',
        'VaR95': total_var_absolute,
        'ES95': total_es_absolute,
        'VaR95_Pct': total_var_pct,
        'ES95_Pct': total_es_pct
    })

    return pd.DataFrame(results)

def main():
    portfolio_df = pd.read_csv('test9_1_portfolio.csv')
    returns_df = pd.read_csv('test9_1_returns.csv')

    results_df = calculate_var_es_copula(
        returns_df,
        portfolio_df,
        alpha=0.05,
        n_simulations=100000,
        random_seed=42
    )

    print(results_df.to_csv(index=False))

if __name__ == "__main__":
    main()
