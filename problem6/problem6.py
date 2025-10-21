import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def calculate_arithmetic_returns(data):
    returns_data = data.copy()
    price_columns = [col for col in data.columns if col not in ['Date']]

    for col in price_columns:
        returns_data[col] = data[col].pct_change()

    returns_data = returns_data.dropna()
    return returns_data


def demean_returns(returns_df):
    demeaned = returns_df.copy()
    price_columns = [col for col in returns_df.columns if col not in ['Date']]

    for col in price_columns:
        demeaned[col] = returns_df[col] - returns_df[col].mean()

    return demeaned


def fit_t_distribution(data):
    params = stats.t.fit(data)
    nu = params[0]
    mu = params[1]
    sigma = params[2]
    return nu, mu, sigma


def gaussian_copula_simulation(returns_df, fitted_params, n_simulations=100000, random_seed=42):
    np.random.seed(random_seed)

    stocks = list(fitted_params.keys())
    n_stocks = len(stocks)

    U = np.zeros((len(returns_df), n_stocks))
    for i, stock in enumerate(stocks):
        returns = returns_df[stock].values
        nu, mu, sigma = fitted_params[stock]
        U[:, i] = stats.t.cdf(returns, df=nu, loc=mu, scale=sigma)

    Z = stats.norm.ppf(U)

    corr_matrix = np.corrcoef(Z.T)

    mean_vec = np.zeros(n_stocks)
    simulated_Z = np.random.multivariate_normal(mean_vec, corr_matrix, size=n_simulations)

    simulated_U = stats.norm.cdf(simulated_Z)

    simulated_returns = np.zeros((n_simulations, n_stocks))
    for i, stock in enumerate(stocks):
        nu, mu, sigma = fitted_params[stock]
        simulated_returns[:, i] = stats.t.ppf(simulated_U[:, i], df=nu, loc=mu, scale=sigma)

    return simulated_returns, corr_matrix


def calculate_var_es_portfolio(simulated_returns, current_prices, holdings, stocks, alpha=0.05):
    n_simulations = simulated_returns.shape[0]
    n_stocks = len(stocks)

    results = []

    total_simulated_values = np.zeros(n_simulations)
    total_current_value = 0

    for i, stock in enumerate(stocks):
        current_price = current_prices[stock]
        holding = holdings[stock]

        current_value = holding * current_price
        total_current_value += current_value

        simulated_prices = current_price * (1 + simulated_returns[:, i])
        simulated_values = holding * simulated_prices
        total_simulated_values += simulated_values

        simulated_pnl = simulated_values - current_value
        sorted_pnl = np.sort(simulated_pnl)

        var_index = int(np.floor(alpha * n_simulations))
        var_dollar = -sorted_pnl[var_index]
        es_tail = sorted_pnl[:var_index]
        es_dollar = -np.mean(es_tail)

        results.append({
            'Stock': stock,
            'Current Value': current_value,
            'VaR (5%)': var_dollar,
            'ES (5%)': es_dollar
        })

    total_pnl = total_simulated_values - total_current_value
    sorted_total_pnl = np.sort(total_pnl)

    var_index = int(np.floor(alpha * n_simulations))
    total_var_dollar = -sorted_total_pnl[var_index]
    total_es_tail = sorted_total_pnl[:var_index]
    total_es_dollar = -np.mean(total_es_tail)

    results.append({
        'Stock': 'Total Portfolio',
        'Current Value': total_current_value,
        'VaR (5%)': total_var_dollar,
        'ES (5%)': total_es_dollar
    })

    return pd.DataFrame(results)


def main():
    df = pd.read_csv('problem6.csv')

    holdings = {'x1': 100, 'x2': 100, 'x3': 100}
    stocks = ['x1', 'x2', 'x3']

    current_prices = {}
    for stock in stocks:
        current_prices[stock] = df[stock].iloc[-1]

    total_value = 0
    for stock in stocks:
        value = holdings[stock] * current_prices[stock]
        total_value += value

    returns_df = calculate_arithmetic_returns(df)

    print("Part a")

    demeaned_returns = demean_returns(returns_df)

    fitted_params = {}
    print(f"{'Stock':>10} {'ν':>15} {'μ':>15} {'σ':>15}")

    for stock in stocks:
        nu, mu, sigma = fit_t_distribution(demeaned_returns[stock].values)
        fitted_params[stock] = (nu, mu, sigma)
        print(f"{stock:>10} {nu:>15.10f} {mu:>15.10f} {sigma:>15.10f}")


    print("-" * 80)
    print("Part b")

    n_simulations = 100000
    simulated_returns, corr_matrix = gaussian_copula_simulation(
        demeaned_returns, fitted_params, n_simulations=n_simulations, random_seed=42
    )

    print(f"{'':>10}", end="")
    for stock in stocks:
        print(f"{stock:>15}", end="")
    print()

    for i, stock_i in enumerate(stocks):
        print(f"{stock_i:>10}", end="")
        for j, stock_j in enumerate(stocks):
            print(f"{corr_matrix[i, j]:>15.10f}", end="")
        print()


    print("-" * 80)
    print("Part c")

    results_df = calculate_var_es_portfolio(
        simulated_returns, current_prices, holdings, stocks, alpha=0.05
    )

    individual_results = results_df[results_df['Stock'] != 'Total Portfolio']

    print(f"{'Stock':>10} {'Current Value':>15} {'VaR':>15} {'ES':>15}")

    for _, row in individual_results.iterrows():
        stock = row['Stock']
        current_val = row['Current Value']
        var_val = row['VaR (5%)']
        es_val = row['ES (5%)']

        print(f"{stock:>10} ${current_val:>14.2f} ${var_val:>14.2f} ${es_val:>14.2f}")

    print("-" * 80)
    print("Part d")

    portfolio_result = results_df[results_df['Stock'] == 'Total Portfolio'].iloc[0]

    print(f"  Current Portfolio Value: ${portfolio_result['Current Value']:.2f}")
    print(f"  VaR at 5%: ${portfolio_result['VaR (5%)']:.2f}")
    print(f"  ES at 5%: ${portfolio_result['ES (5%)']:.2f}")

if __name__ == "__main__":
    main()


