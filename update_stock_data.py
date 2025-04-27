import yfinance as yf
import pandas as pd

# Ticker symbols and the desired column names
tickers = {
    'AMZN': 'amzn',
    'DPZ': 'dpz',
    'BTC-USD': 'btc',
    'NFLX': 'nflx'
}

# Date range
start_date = '2021-01-01'
end_date = '2025-04-01'

# Dictionary to hold each DataFrame
data_frames = {}

# Download and rename Close columns
for ticker, col_name in tickers.items():
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    df.rename(columns={'Close': col_name}, inplace=True)
    data_frames[col_name] = df

# Merge all DataFrames on the index (Date) using inner join to keep only common dates
combined_df = data_frames['amzn']
for col in ['dpz', 'btc', 'nflx']:
    combined_df = combined_df.join(data_frames[col], how='inner')

# Reset index to move Date into a column
combined_df.reset_index(inplace=True)
combined_df.rename(columns={'Date': 'date'}, inplace=True)

# Save the result
combined_df.to_csv("combined_closing_prices.csv", index=False)

print("✅ Combined CSV saved as 'combined_closing_prices.csv'.")