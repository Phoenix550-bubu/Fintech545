import pandas as pd
import numpy as np


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

    # Total Return Calculation

    total_returns = {}

    for i, asset in enumerate(asset_names):
        asset_returns = returns_matrix[:, i]
        total_ret = np.exp(np.sum(np.log(1.0 + asset_returns))) - 1
        total_returns[asset] = total_ret

    portfolio_total_return = np.exp(np.sum(np.log(1.0 + portfolio_returns))) - 1
    total_returns['Portfolio'] = portfolio_total_return

    # Return Attribution using Carino K
    K = np.log(1 + portfolio_total_return) / portfolio_total_return

    carino_k = np.log(1.0 + portfolio_returns) / portfolio_returns / K

    return_attribution = {}

    for i, asset in enumerate(asset_names):
        asset_returns = returns_matrix[:, i]
        weights_over_time = weights_matrix[:, i]
        attr = np.sum(asset_returns * weights_over_time * carino_k)
        return_attribution[asset] = attr

    return_attribution['Portfolio'] = portfolio_total_return

    # Volatility Attribution
    Y = returns_matrix * weights_matrix

    X = np.column_stack([np.ones(len(portfolio_returns)), portfolio_returns])

    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    beta = B[1, :]

    portfolio_std = np.std(portfolio_returns, ddof=1)
    component_sd = beta * portfolio_std

    vol_attribution = {}
    for i, asset in enumerate(asset_names):
        vol_attribution[asset] = component_sd[i]

    vol_attribution['Portfolio'] = portfolio_std

    # Create Output DataFrame
    attribution_df = pd.DataFrame({
        'Value': ['TotalReturn', 'Return Attribution', 'Vol Attribution']
    })

    for asset in asset_names:
        attribution_df[asset] = [
            total_returns[asset],
            return_attribution[asset],
            vol_attribution[asset]
        ]

    attribution_df['Portfolio'] = [
        total_returns['Portfolio'],
        return_attribution['Portfolio'],
        vol_attribution['Portfolio']
    ]

    return attribution_df


def main():
    returns_df = pd.read_csv('test11_1_returns.csv')

    weights_df = pd.read_csv('test11_1_weights.csv')
    weights = weights_df['W'].values

    attribution = expost_attribution(returns_df, weights)

    print(attribution.to_csv(index=False))


if __name__ == "__main__":
    main()
