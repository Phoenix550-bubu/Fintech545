import pandas as pd
import numpy as np


def expost_factor_attribution(stock_returns, factor_returns, betas, initial_weights):
    stock_names = stock_returns.columns.tolist()
    factor_names = factor_returns.columns.tolist()
    n_stocks = len(stock_names)
    n_factors = len(factor_names)

    stock_returns_matrix = stock_returns.values
    factor_returns_matrix = factor_returns.values
    n_periods = len(stock_returns_matrix)

    beta_matrix = betas[factor_names].values

    portfolio_returns = np.zeros(n_periods)
    residual_returns = np.zeros(n_periods)
    weights_matrix = np.zeros((n_periods, n_stocks))
    factor_weights_matrix = np.zeros((n_periods, n_factors))

    current_weights = initial_weights.copy()

    for i in range(n_periods):
        weights_matrix[i, :] = current_weights

        factor_weights = beta_matrix.T @ current_weights
        factor_weights_matrix[i, :] = factor_weights

        stock_rets = stock_returns_matrix[i, :]
        current_weights = current_weights * (1.0 + stock_rets)

        pR = np.sum(current_weights)

        current_weights = current_weights / pR

        portfolio_returns[i] = pR - 1

        factor_rets = factor_returns_matrix[i, :]
        residual_returns[i] = (pR - 1) - factor_weights @ factor_rets

    # Total Return Calculation

    total_returns = {}

    for i, factor in enumerate(factor_names):
        factor_rets = factor_returns_matrix[:, i]
        total_ret = np.exp(np.sum(np.log(1.0 + factor_rets))) - 1
        total_returns[factor] = total_ret

    alpha_total_return = np.exp(np.sum(np.log(1.0 + residual_returns))) - 1
    total_returns['Alpha'] = alpha_total_return

    # Total return for portfolio
    portfolio_total_return = np.exp(np.sum(np.log(1.0 + portfolio_returns))) - 1
    total_returns['Portfolio'] = portfolio_total_return

    # Return Attribution using Carino K

    K = np.log(1 + portfolio_total_return) / portfolio_total_return

    carino_k = np.log(1.0 + portfolio_returns) / portfolio_returns / K

    return_attribution = {}

    for i, factor in enumerate(factor_names):
        factor_rets = factor_returns_matrix[:, i]
        factor_weights = factor_weights_matrix[:, i]
        attr = np.sum(factor_rets * factor_weights * carino_k)
        return_attribution[factor] = attr

    alpha_attr = np.sum(residual_returns * carino_k)
    return_attribution['Alpha'] = alpha_attr

    return_attribution['Portfolio'] = portfolio_total_return

    # Volatility Attribution

    Y = np.column_stack([
        factor_returns_matrix * factor_weights_matrix,
        residual_returns
    ])

    X = np.column_stack([np.ones(len(portfolio_returns)), portfolio_returns])

    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    beta = B[1, :]

    portfolio_std = np.std(portfolio_returns, ddof=1)
    component_sd = beta * portfolio_std

    vol_attribution = {}
    for i, factor in enumerate(factor_names):
        vol_attribution[factor] = component_sd[i]

    vol_attribution['Alpha'] = component_sd[n_factors]
    vol_attribution['Portfolio'] = portfolio_std

    # Create Output DataFrame

    attribution_df = pd.DataFrame({
        'Value': ['TotalReturn', 'Return Attribution', 'Vol Attribution']
    })

    for factor in factor_names:
        attribution_df[factor] = [
            total_returns[factor],
            return_attribution[factor],
            vol_attribution[factor]
        ]

    attribution_df['Alpha'] = [
        total_returns['Alpha'],
        return_attribution['Alpha'],
        vol_attribution['Alpha']
    ]

    attribution_df['Portfolio'] = [
        total_returns['Portfolio'],
        return_attribution['Portfolio'],
        vol_attribution['Portfolio']
    ]

    return attribution_df


def main():
    stock_returns = pd.read_csv('test11_2_stock_returns.csv')
    factor_returns = pd.read_csv('test11_2_factor_returns.csv')
    betas = pd.read_csv('test11_2_beta.csv')
    weights_df = pd.read_csv('test11_2_weights.csv')
    weights = weights_df['W'].values

    attribution = expost_factor_attribution(stock_returns, factor_returns, betas, weights)

    print(attribution.to_csv(index=False))


if __name__ == "__main__":
    main()
