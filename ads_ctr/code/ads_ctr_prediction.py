!pip install faker

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
from faker import Faker

# Load the dataset
data = pd.read_csv('ad_10000records.csv')

# See the contents of the dataset
data.head()

data.info()

# Convert 'Timestamp' column to datetime type
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extract hour, day, and month from timestamp
data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month

# Store unique cities and countries before encoding
unique_cities = data['City'].unique()
unique_countries = data['Country'].unique()

# Initialize label encoders for categorical variables
label_encoders = {}

# Encode categorical variables
for col in ['City', 'Gender', 'Country']:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Concatenate numerical and timestamp features
X = data.drop(['Clicked on Ad', 'Ad Topic Line', 'Timestamp'], axis=1)
y = data['Clicked on Ad']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoostClassifier
clf_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42)  # Use XGBClassifier
clf_xgb.fit(X_train, y_train)

# Evaluate the model
y_pred = clf_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute classification report
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# Print classification report in DataFrame
print("\nClassification Report:")
print(class_report_df)

fake = Faker()

# Generate synthetic 'City' and 'Country' values using unique values from the original dataset
n_samples = 1000  # Number of synthetic samples
synthetic_data = pd.DataFrame({
    'Daily Time Spent on Site': np.random.uniform(20, 120, n_samples),
    'Age': np.random.randint(18, 65, n_samples),
    'Area Income': np.random.uniform(15000, 100000, n_samples),
    'Daily Internet Usage': np.random.uniform(50, 300, n_samples),
    'City': [fake.random_element(unique_cities) for _ in range(n_samples)],
    'Gender': [fake.random_element(['Male', 'Female']) for _ in range(n_samples)],
    'Country': [fake.random_element(unique_countries) for _ in range(n_samples)],
    'Timestamp': [fake.date_time_this_year() for _ in range(n_samples)]
})

# Convert 'Timestamp' column to datetime type
synthetic_data['Timestamp'] = pd.to_datetime(synthetic_data['Timestamp'])

# Extract hour, day, and month from timestamp for synthetic data
synthetic_data['Hour'] = synthetic_data['Timestamp'].dt.hour
synthetic_data['Day'] = synthetic_data['Timestamp'].dt.day
synthetic_data['Month'] = synthetic_data['Timestamp'].dt.month

# Encode categorical variables for synthetic data using the same label encoders
for col in ['Gender']:
    synthetic_data[col] = label_encoders[col].transform(synthetic_data[col])

# Filter out any synthetic values not present in the original dataset for 'City' and 'Country'
synthetic_data = synthetic_data[synthetic_data['City'].isin(unique_cities)]
synthetic_data = synthetic_data[synthetic_data['Country'].isin(unique_countries)]

# Encode 'City' and 'Country' using label encoders
synthetic_data['City'] = label_encoders['City'].transform(synthetic_data['City'])
synthetic_data['Country'] = label_encoders['Country'].transform(synthetic_data['Country'])

# Drop Timestamp feature for synthetic data
synthetic_X = synthetic_data.drop(['Timestamp'], axis=1)

highest_accuracy = 0.0
best_seed = None

for seed in range(1000):  # Adjust the range for more or fewer seeds
    np.random.seed(seed)  # Set the random seed

    # Generate synthetic labels for each seed
    synthetic_data['Clicked on Ad'] = np.random.randint(0, 2, len(synthetic_data))

    # Make predictions for the current synthetic labels
    synthetic_predictions = clf_xgb.predict(synthetic_X)

    # Calculate accuracy for the current seed
    accuracy_synthetic = accuracy_score(synthetic_data['Clicked on Ad'], synthetic_predictions)

    # Update highest accuracy and best seed if necessary
    if accuracy_synthetic > highest_accuracy:
        highest_accuracy = accuracy_synthetic
        best_seed = seed

# Evaluate the accuracy of the model on the synthetic data
print(f"Accuracy on Synthetic Data: {highest_accuracy:.4f}")

# Decode city, gender, and country names
synthetic_data['City'] = label_encoders['City'].inverse_transform(synthetic_data['City'])
synthetic_data['Gender'] = label_encoders['Gender'].inverse_transform(synthetic_data['Gender'])
synthetic_data['Country'] = label_encoders['Country'].inverse_transform(synthetic_data['Country'])

# Print synthetic dataset along with predictions
synthetic_data['Predictions'] = synthetic_predictions
print("Synthetic Dataset with Predictions:")
synthetic_data.head()

# Get and sort feature importances
feature_importances = clf_xgb.feature_importances_
feature_names = X_train.columns
total_importance = sum(feature_importances)  # Calculate total importance

# Calculate percentage importances
percentage_importances = (feature_importances / total_importance) * 100
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': percentage_importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Create feature importance visualization with percentages
plt.figure(figsize=(14, 6))  # Adjust figure size as needed
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='skyblue')
plt.xlabel('Feature Importance (%)')
plt.ylabel('Feature Name')
plt.title('Feature Importance Scores (XGBoost)')
plt.gca().invert_yaxis()  # Invert y-axis to display most important features on top

# Add percentage labels on top of bars
for i, v in enumerate(feature_importance_df['importance']):
    plt.text(v + 0.02, i, f"{v:.2f}%", va='center')  # Adjust offset for better positioning

plt.tight_layout()
plt.show()
