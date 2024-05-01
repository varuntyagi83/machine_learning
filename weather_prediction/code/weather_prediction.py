import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read the CSV file
df = pd.read_csv('3599883.csv')

# Feature engineering and data preprocessing
df['DATE'] = pd.to_datetime(df['DATE'])  # Convert DATE to datetime format
df.set_index('DATE', inplace=True)  # Set DATE as the index
df.sort_index(inplace=True)  # Sort by index

# Create target variable (average temperature)
df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2

# Handle missing values appropriately
df = df.dropna(subset=['TMAX', 'TMIN', 'TAVG', 'PRCP', 'LATITUDE', 'LONGITUDE'])

# Create input and target variables
X = df[['TMAX', 'TMIN', 'PRCP', 'LATITUDE', 'LONGITUDE']]  # Input features
y = df['TAVG']  # Target variable (average temperature)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(8, input_dim=5, activation='relu'))  # Adjust number of neurons if needed
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

# Save the trained model
model.save('saved_model.h5')  # Replace 'saved_model.h5' with your desired filename
print("Model saved successfully!")
