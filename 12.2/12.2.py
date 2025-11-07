import pandas as pd
import numpy as np


def bt_american(call: bool, underlying: float, strike: float, ttm: float,
                rf: float, b: float, ivol: float, N: int) -> float:
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    def n_node_func(n):
        return int((n + 1) * (n + 2) / 2)

    def idx_func(i, j):
        return n_node_func(j - 1) + i

    n_nodes = n_node_func(N)
    option_values = np.zeros(n_nodes)

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idx_func(i, j)
            price = underlying * (u ** i) * (d ** (j - i))

            option_values[idx] = max(0, z * (price - strike))

            if j < N:
                continuation_value = df * (pu * option_values[idx_func(i + 1, j + 1)] +
                                          pd * option_values[idx_func(i, j + 1)])
                option_values[idx] = max(option_values[idx], continuation_value)

    return option_values[0]


def calculate_greeks_american(call: bool, underlying: float, strike: float,
                              ttm: float, rf: float, b: float, ivol: float,
                              N: int = 500) -> dict:
    value = bt_american(call, underlying, strike, ttm, rf, b, ivol, N)

    # Delta
    dS_delta = 0.5
    value_up_S = bt_american(call, underlying + dS_delta, strike, ttm, rf, b, ivol, N)
    value_down_S = bt_american(call, underlying - dS_delta, strike, ttm, rf, b, ivol, N)
    delta = (value_up_S - value_down_S) / (2 * dS_delta)

    # Gamma
    dS_gamma = 1.5
    value_up_S_gamma = bt_american(call, underlying + dS_gamma, strike, ttm, rf, b, ivol, N)
    value_down_S_gamma = bt_american(call, underlying - dS_gamma, strike, ttm, rf, b, ivol, N)
    gamma = (value_up_S_gamma + value_down_S_gamma - 2 * value) / (dS_gamma ** 2)

    # Vega
    dvol = 0.01
    value_up_vol = bt_american(call, underlying, strike, ttm, rf, b, ivol + dvol, N)
    value_down_vol = bt_american(call, underlying, strike, ttm, rf, b, ivol - dvol, N)
    vega = (value_up_vol - value_down_vol) / (2 * dvol)

    # Rho
    dr = 0.01
    value_up_r = bt_american(call, underlying, strike, ttm, rf + dr, b, ivol, N)
    value_down_r = bt_american(call, underlying, strike, ttm, rf - dr, b, ivol, N)
    rho = (value_up_r - value_down_r) / (2 * dr)

    # Theta
    dt = 0.5 / 365
    if ttm > dt:
        value_minus_dt = bt_american(call, underlying, strike, ttm - dt, rf, b, ivol, N)
        theta = (value - value_minus_dt) / dt
    else:
        theta = 0.0

    return {
        'value': value,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta
    }


def main():
    input_file = "test12_1.csv"

    df = pd.read_csv(input_file)

    df = df.dropna(subset=['ID'])

    results = []

    for idx, row in df.iterrows():
        option_id = int(row['ID'])
        option_type = row['Option Type'].strip()
        underlying = float(row['Underlying'])
        strike = float(row['Strike'])
        days_to_maturity = float(row['DaysToMaturity'])
        days_per_year = float(row['DayPerYear'])
        rf = float(row['RiskFreeRate'])
        dividend_rate = float(row['DividendRate'])
        ivol = float(row['ImpliedVol'])

        ttm = days_to_maturity / days_per_year

        b = rf - dividend_rate

        is_call = (option_type.lower() == 'call')

        greeks = calculate_greeks_american(is_call, underlying, strike, ttm, rf, b, ivol, N=500)

        results.append({
            'ID': option_id,
            'Value': greeks['value'],
            'Delta': greeks['delta'],
            'Gamma': greeks['gamma'],
            'Vega': greeks['vega'],
            'Rho': greeks['rho'],
            'Theta': greeks['theta']
        })

    output_df = pd.DataFrame(results)

    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()
