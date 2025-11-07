import pandas as pd
import numpy as np
from typing import List


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


def bt_american_discrete_div(call: bool, underlying: float, strike: float, ttm: float,
                              rf: float, div_amts: List[float], div_times: List[int],
                              ivol: float, N: int) -> float:
    if len(div_amts) == 0 or len(div_times) == 0:
        return bt_american(call, underlying, strike, ttm, rf, rf, ivol, N)

    if div_times[0] > N:
        return bt_american(call, underlying, strike, ttm, rf, rf, ivol, N)

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    def n_node_func(n):
        return int((n + 1) * (n + 2) / 2)

    def idx_func(i, j):
        return n_node_func(j - 1) + i

    n_div = len(div_times)
    n_nodes = n_node_func(div_times[0])
    option_values = np.zeros(n_nodes)

    for j in range(div_times[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idx_func(i, j)
            price = underlying * (u ** i) * (d ** (j - i))

            if j < div_times[0]:
                option_values[idx] = max(0, z * (price - strike))
                continuation_value = df * (pu * option_values[idx_func(i + 1, j + 1)] +
                                          pd * option_values[idx_func(i, j + 1)])
                option_values[idx] = max(option_values[idx], continuation_value)
            else:
                val_exercise = max(0, z * (price - strike))

                remaining_div_amts = div_amts[1:n_div]
                remaining_div_times = [t - div_times[0] for t in div_times[1:n_div]]

                val_no_exercise = bt_american_discrete_div(
                    call,
                    price - div_amts[0],
                    strike,
                    ttm - div_times[0] * dt,
                    rf,
                    remaining_div_amts,
                    remaining_div_times,
                    ivol,
                    N - div_times[0]
                )

                option_values[idx] = max(val_no_exercise, val_exercise)

    return option_values[0]


def main():
    input_file = "test12_3.csv"

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
        ivol = float(row['ImpliedVol'])

        div_dates_str = row['DividendDates'].strip()
        div_amts_str = row['DividendAmts'].strip()

        div_dates = [int(d) for d in div_dates_str.split(',')]
        div_amts = [float(a) for a in div_amts_str.split(',')]

        ttm = days_to_maturity / days_per_year
        is_call = (option_type.lower() == 'call')

        N = 300

        div_times = [int(d * N / days_to_maturity) for d in div_dates]

        value = bt_american_discrete_div(is_call, underlying, strike, ttm, rf,
                                         div_amts, div_times, ivol, N)

        results.append({
            'ID': option_id,
            'Value': value
        })

    output_df = pd.DataFrame(results)
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()
