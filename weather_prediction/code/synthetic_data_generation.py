import pandas as pd
from scipy.stats import norm, gamma, uniform
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('saved_model.h5')  # Replace with the filename you used

# Define parameters for data generation
num_samples = 1000  # Adjust as needed
lat_min, lat_max = -90, 90  # Example latitude range
lon_min, lon_max = -180, 180  # Example longitude range

# Generate random values from chosen distributions
tmax = norm.rvs(loc=20, scale=5, size=num_samples)
tmin = norm.rvs(loc=15, scale=4, size=num_samples)
prcp = gamma.rvs(a=2, scale=0.5, size=num_samples)
latitude = uniform.rvs(loc=lat_min, scale=lat_max - lat_min, size=num_samples)
longitude = uniform.rvs(loc=lon_min, scale=lon_max - lon_min, size=num_samples)

# Calculate average temperature
tavg = (tmax + tmin) / 2

# Create the DataFrame
data = {'TMAX': tmax, 'TMIN': tmin, 'PRCP': prcp,
        'LATITUDE': latitude, 'LONGITUDE': longitude}
df_test = pd.DataFrame(data)

# Standardize the synthetic data using the same scaler
df_test_scaled = scaler.transform(df_test)

# Make predictions
y_test_pred = model.predict(df_test_scaled)

# Print the results
print("Predicted average temperatures:")
print(y_test_pred)
