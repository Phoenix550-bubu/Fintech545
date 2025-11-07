import pandas as pd
import numpy as np
from scipy.stats import norm


def gbsm(call: bool, underlying: float, strike: float, ttm: float,
         rf: float, b: float, ivol: float) -> dict:
    if ttm <= 0:
        if call:
            value = max(0, underlying - strike)
        else:
            value = max(0, strike - underlying)
        return {
            'value': value,
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'theta': 0.0
        }

    d1 = (np.log(underlying / strike) + (b + ivol**2 / 2) * ttm) / (ivol * np.sqrt(ttm))
    d2 = d1 - ivol * np.sqrt(ttm)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)
    n_d1 = norm.pdf(d1)

    if call:
        value = underlying * np.exp((b - rf) * ttm) * N_d1 - strike * np.exp(-rf * ttm) * N_d2
    else:
        value = strike * np.exp(-rf * ttm) * N_neg_d2 - underlying * np.exp((b - rf) * ttm) * N_neg_d1

    # Delta
    if call:
        delta = np.exp((b - rf) * ttm) * N_d1
    else:
        delta = np.exp((b - rf) * ttm) * (N_d1 - 1)

    # Gamma
    gamma = n_d1 * np.exp((b - rf) * ttm) / (underlying * ivol * np.sqrt(ttm))

    # Vega
    vega = underlying * np.exp((b - rf) * ttm) * n_d1 * np.sqrt(ttm)

    # Theta
    if call:
        theta = (-(underlying * np.exp((b - rf) * ttm) * n_d1 * ivol / (2 * np.sqrt(ttm)))
                 - (b - rf) * underlying * np.exp((b - rf) * ttm) * N_d1
                 - rf * strike * np.exp(-rf * ttm) * N_d2)
    else:
        theta = (-(underlying * np.exp((b - rf) * ttm) * n_d1 * ivol / (2 * np.sqrt(ttm)))
                 + (b - rf) * underlying * np.exp((b - rf) * ttm) * N_neg_d1
                 + rf * strike * np.exp(-rf * ttm) * N_neg_d2)

    # Rho
    if call:
        rho = ttm * strike * np.exp(-rf * ttm) * N_d2
    else:
        rho = -ttm * strike * np.exp(-rf * ttm) * N_neg_d2

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

        greeks = gbsm(is_call, underlying, strike, ttm, rf, b, ivol)

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
