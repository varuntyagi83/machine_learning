import matplotlib.pyplot as plt
import yfinance as yf

# Download stock data
ticker = 'GOOGL'
start_date = '2020-01-01'
end_date = '2024-02-25'
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the DataFrame
print(stock_data.head())

# Plotting the closing prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Adj Close'], label='Adjusted Close Price', color='blue')
plt.title(f'{ticker} Stock Prices from {start_date} to {end_date}')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()
