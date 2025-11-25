import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt


def gbsm(call, underlying, strike, ttm, rf, b, ivol):
    d1 = (np.log(underlying / strike) + (b + ivol**2 / 2) * ttm) / (ivol * np.sqrt(ttm))
    d2 = d1 - ivol * np.sqrt(ttm)

    if call:
        value = underlying * np.exp((b - rf) * ttm) * norm.cdf(d1) - strike * np.exp(-rf * ttm) * norm.cdf(d2)
    else:
        value = strike * np.exp(-rf * ttm) * norm.cdf(-d2) - underlying * np.exp((b - rf) * ttm) * norm.cdf(-d1)

    return value


def calculate_option_price_from_simulation(returns, strike, S0, rf, ttm):
    terminal_prices = S0 * (1 + returns)

    call_payoffs = np.maximum(terminal_prices - strike, 0)
    put_payoffs = np.maximum(strike - terminal_prices, 0)

    discount_factor = np.exp(-rf * ttm)
    call_price = discount_factor * np.mean(call_payoffs)
    put_price = discount_factor * np.mean(put_payoffs)

    return call_price, put_price


def implied_volatility(option_price, call, underlying, strike, ttm, rf, b):
    def objective(ivol):
        return gbsm(call, underlying, strike, ttm, rf, b, ivol) - option_price

    try:
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
        return iv
    except:
        return np.nan


def main():
    df = pd.read_csv('problem1.csv')
    returns = df['r'].values

    S0 = 100.0
    rf = 0.04
    ttm = 1 / 255
    b = rf

    print("=" * 80)
    print("Part a: Option Prices for Strikes")

    strikes_a = [99, 100, 101]

    for strike in strikes_a:
        call_price, put_price = calculate_option_price_from_simulation(returns, strike, S0, rf, ttm)
        print(f"\nStrike: {strike}")
        print(f"  Call Price: {call_price:.10f}")
        print(f"  Put Price:  {put_price:.10f}")

    print("\n" + "=" * 80)
    print("Part b: Implied Volatility Smile")

    strikes_b = np.arange(95, 106, 1)
    implied_vols = []

    print(f"\n{'Strike':>10} {'Call Price':>15} {'Implied Vol':>15}")

    for strike in strikes_b:
        call_price, _ = calculate_option_price_from_simulation(returns, strike, S0, rf, ttm)

        iv = implied_volatility(call_price, True, S0, strike, ttm, rf, b)
        implied_vols.append(iv)

        print(f"{strike:>10} {call_price:>15.10f} {iv:>15.10f}")

    plt.figure(figsize=(10, 6))
    plt.plot(strikes_b, implied_vols, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Strike Price', fontsize=12)
    plt.ylabel('Implied Volatility', fontsize=12)
    plt.title('Implied Volatility Smile', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=100, color='r', linestyle='--', alpha=0.5, label='ATM Strike')
    plt.legend()
    plt.tight_layout()
    plt.savefig('implied_volatility_smile.png', dpi=300)


if __name__ == "__main__":
    main()
