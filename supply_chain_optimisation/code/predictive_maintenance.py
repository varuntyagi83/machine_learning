import cv2
import numpy as np

# Define a function to read sensor data from a file
def read_sensor_data(file_path):
    # Open the sensor data file
    with open(file_path, 'r') as f:
        sensor_data = f.readlines()

    # Convert sensor data to a NumPy array
    sensor_data = np.array([float(line.strip()) for line in sensor_data])

    return sensor_data

# Define a function to train a linear regression model
def train_linear_regression_model(sensor_data):
    # Split the sensor data into training and testing sets
    train_data = sensor_data[:int(0.8 * len(sensor_data))]
    test_data = sensor_data[int(0.8 * len(sensor_data)):]

    # Train a linear regression model on the training data
    model = LinearRegression()
    model.fit(train_data.reshape(-1, 1), train_data)

    return model

# Define a function to predict sensor failure
def predict_sensor_failure(model, sensor_data):
    # Predict the sensor's remaining useful life
    remaining_life = model.predict(sensor_data.reshape(-1, 1))

    # Check if the sensor is likely to fail
    if remaining_life < 0:
        print("Sensor failure is likely")
    else:
        print("Sensor failure is unlikely")

# Read sensor data from a file
sensor_data = read_sensor_data('sensor_data.txt')

# Train a linear regression model
model = train_linear_regression_model(sensor_data)

# Predict sensor failure
predict_sensor_failure(model, sensor_data[-1])
