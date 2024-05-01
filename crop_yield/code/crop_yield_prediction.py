from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
import joblib
import random
import pandas as pd
import numpy as np

# File path to the downloaded dataset
file_path = "yield_df.csv"

# Load the yield dataset
yield_df = pd.read_csv(file_path, delimiter=",")

# Display the first few rows of the DataFrame
yield_df.head()

# Dropping unnecessary column
yield_df.drop('Unnamed: 0', axis=1, inplace=True)

# Removing countries with less than 100 records
country_counts = yield_df['Area'].value_counts()
countries_to_keep = country_counts[country_counts >= 100].index
yield_df = yield_df[yield_df['Area'].isin(countries_to_keep)]

# Removing outliers
yield_df = yield_df[yield_df['hg/ha_yield'] > 0]  # Remove rows with negative yield values

# Split data into features and target variable
X = yield_df[['Area', 'Item', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']].copy()
y = yield_df['hg/ha_yield']

# Use LabelEncoder for categorical variables
label_encoder_area = LabelEncoder()
label_encoder_item = LabelEncoder()
X['Area'] = label_encoder_area.fit_transform(X['Area'])
X['Item'] = label_encoder_item.fit_transform(X['Item'])

# Use StandardScaler for numerical variables
scaler = StandardScaler()
numerical_columns = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize individual models
model_lr = LinearRegression()
model_rf = RandomForestRegressor()
model_gb = GradientBoostingRegressor()
model_xgb = xgb.XGBRegressor()
model_knn = KNeighborsRegressor()
model_dt = DecisionTreeRegressor()
model_bagging = BaggingRegressor()

# Fit the models on training data
models = [model_lr, model_rf, model_gb, model_xgb, model_knn, model_dt, model_bagging]
model_names = ['Linear Regression', 'Random Forest', 'Gradient Boost', 'XGBoost', 'KNN', 'Decision Tree', 'Bagging Regressor']
metrics_data = []

for model, model_name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test) * 100
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) * 100
    metrics_data.append({'Model': model_name, 'Accuracy (%)': accuracy, 'MSE': mse, 'R^2 (%)': r2})

# Create dataframe from metrics data
metrics_df = pd.DataFrame(metrics_data)

# Print metrics for individual models
print(metrics_df)

# Create an ensemble of models
ensemble_model = VotingRegressor(estimators=[
    ('lr', model_lr),
    ('rf', model_rf),
    ('gb', model_gb),
    ('xgb', model_xgb),
    ('knn', model_knn),
    ('dt', model_dt),
    ('bagging', model_bagging)
])

# Fit the ensemble model on training data
ensemble_model.fit(X_train, y_train)

# Save the ensemble model
joblib.dump(ensemble_model, 'ensemble_model.pkl')

# Save the label encoder for Area
joblib.dump(label_encoder_area, 'label_encoder_area.pkl')

# Save the label encoder for Item
joblib.dump(label_encoder_item, 'label_encoder_item.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Dictionary to store mean and range for each numerical feature for each country
country_stats = {}

# Group original dataset by country
grouped_yield_df = yield_df.groupby('Area')

# Calculate mean and range for each numerical feature for each country
for country, group in grouped_yield_df:
    country_stats[country] = {
        'avg_temp_mean': group['avg_temp'].mean(),
        'avg_temp_range': group['avg_temp'].max() - group['avg_temp'].min(),
        'rainfall_mean': group['average_rain_fall_mm_per_year'].mean(),
        'rainfall_range': group['average_rain_fall_mm_per_year'].max() - group['average_rain_fall_mm_per_year'].min(),
        'pesticides_mean': group['pesticides_tonnes'].mean(),
        'pesticides_range': group['pesticides_tonnes'].max() - group['pesticides_tonnes'].min()
    }

# Generate synthetic data for each country within its specific range
synthetic_data = pd.DataFrame()

for country, stats in country_stats.items():
    # Get the list of crops planted in the current country
    crops_in_country = set(yield_df[yield_df['Area'] == country]['Item'])
    
    num_samples = 1000  # Number of synthetic samples per country
    
    # Generate synthetic data for current country and crops
    random_data = {
        'Area': [country] * num_samples,
        'Item': np.random.choice(list(crops_in_country), size=num_samples),
        'average_rain_fall_mm_per_year': np.round(np.random.uniform(stats['rainfall_mean'] - 0.1 * stats['rainfall_range'],
                                                                      stats['rainfall_mean'] + 0.1 * stats['rainfall_range'],
                                                                      size=num_samples)),
        'pesticides_tonnes': np.round(np.random.uniform(stats['pesticides_mean'] - 0.1 * stats['pesticides_range'],
                                                         stats['pesticides_mean'] + 0.1 * stats['pesticides_range'],
                                                         size=num_samples)),
        'avg_temp': np.round(np.random.uniform(stats['avg_temp_mean'] - 0.1 * stats['avg_temp_range'],
                                                stats['avg_temp_mean'] + 0.1 * stats['avg_temp_range'],
                                                size=num_samples))
    }
    
    # Convert numerical features to integers
    random_data['average_rain_fall_mm_per_year'] = random_data['average_rain_fall_mm_per_year'].astype(int)
    random_data['pesticides_tonnes'] = random_data['pesticides_tonnes'].astype(int)
    random_data['avg_temp'] = random_data['avg_temp'].astype(int)
    
    # Create DataFrame for current country's synthetic data
    synthetic_data = pd.concat([synthetic_data, pd.DataFrame(random_data)])

# Load the label encoder for 'Area'
label_encoder_area = joblib.load('label_encoder_area.pkl')

# Load the label encoder for 'Item'
label_encoder_item = joblib.load('label_encoder_item.pkl')

# Encode 'Area' variable in the synthetic dataset using the loaded label encoder
synthetic_data['Area'] = label_encoder_area.transform(synthetic_data['Area'])

# Encode 'Item' variable in the synthetic dataset using the loaded label encoder
synthetic_data['Item'] = label_encoder_item.transform(synthetic_data['Item'])

# Scale numerical features of synthetic data using the previously fitted scaler
synthetic_data[numerical_columns] = scaler.transform(synthetic_data[numerical_columns])

# Load the ensemble model
ensemble_model = joblib.load('ensemble_model.pkl')

# Make predictions on synthetic dataset
synthetic_predictions = ensemble_model.predict(synthetic_data)

# Round the predicted yield to one decimal place
synthetic_predictions_rounded = np.round(synthetic_predictions, 1)

# Decode the encoded country and crop labels
synthetic_data['Area'] = label_encoder_area.inverse_transform(synthetic_data['Area'])
synthetic_data['Item'] = label_encoder_item.inverse_transform(synthetic_data['Item'])

# Inverse transform scaled numerical features back to their original values
synthetic_data[numerical_columns] = scaler.inverse_transform(synthetic_data[numerical_columns])

# Convert numerical features to integers for final output
synthetic_data[numerical_columns] = synthetic_data[numerical_columns].astype(int)

# Create DataFrame to store output
output_df = synthetic_data.copy()
output_df['Prediction'] = synthetic_predictions_rounded  # Use rounded predictions

# Shuffle the rows of the DataFrame
output_df = output_df.sample(frac=1).reset_index(drop=True)

# Show the values
output_df.head(20)
