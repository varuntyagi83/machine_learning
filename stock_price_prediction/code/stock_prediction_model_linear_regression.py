import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

# Download stock price data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Adj Close']

# Preprocess data and create input sequences
def preprocess_data(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length].values
        target = data[i+sequence_length]
        sequences.append((sequence, target))
    return sequences

# Split data into training and testing sets
def split_data(data, test_size=0.2):
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    return train_data, test_data

# Train linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate model on test data
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Download stock data
ticker = 'GOOGL'
start_date = '2020-01-01'
end_date = '2024-02-25'
stock_data = download_stock_data(ticker, start_date, end_date)

# Preprocess data
sequence_length = 10
data_sequences = preprocess_data(stock_data, sequence_length)

# Split data into training and testing sets
train_data, test_data = split_data(data_sequences)

# Prepare training data
X_train = np.array([item[0] for item in train_data])
y_train = np.array([item[1] for item in train_data])

# Prepare testing data
X_test = np.array([item[0] for item in test_data])
y_test = np.array([item[1] for item in test_data])

# Train linear regression model
model = train_model(X_train, y_train)

# Example of using the trained model for prediction
last_sequence = X_test[-1].reshape(1, -1)
predicted_price = model.predict(last_sequence)[0]
print(f'\nPredicted Stock Price: {predicted_price: 0.2f}')

# Evaluate model using Mean Squared Error
mse = evaluate_model(model, X_test, y_test)

# Evaluate model using R-squared
r2 = r2_score(y_test, model.predict(X_test))

# Calculate Adjusted R-squared (adjusts for number of features)
n = X_test.shape[0]  # Number of samples
p = X_test.shape[1]  # Number of features

adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Print the R-squared and adjusted R-squared
print(f'R-squared: {r2:.4f}')
print(f'Adjusted R-squared: {adjusted_r2:.2f}')
print(f'Mean Squared Error on Test Data: {mse: 0.2f}')
