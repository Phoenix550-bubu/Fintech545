import pandas as pd
import numpy as np

def calculate_log_returns(data):
    returns_data = data.copy()

    price_columns = [col for col in data.columns if col != 'Date']

    for col in price_columns:
        log_prices = np.log(data[col])
        returns_data[col] = log_prices.diff()

    returns_data = returns_data.dropna()

    return returns_data

def main():
    data = pd.read_csv('test6.csv')

    returns = calculate_log_returns(data)

    print(','.join(returns.columns))

    for _, row in returns.iterrows():
        print(','.join(map(str, row.values)))

if __name__ == "__main__":
    main()