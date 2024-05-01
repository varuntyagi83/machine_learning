# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.datasets import fetch_california_housing

# Download the California Housing dataset
california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

# Median house value for California districts
data['MEDV'] = california_housing.target  

# Print the first 10 rows of the dataset
data.head(10)

# Add more random descriptions
additional_descriptions = [
    "Spacious family home with a beautiful view",
    "Elegant townhouse in a historic district",
    "Luxurious penthouse with top-notch amenities",
    "Quaint cottage surrounded by nature",
    "Contemporary condo with cutting-edge design",
    "Sunny apartment in a lively urban neighborhood",
    "Rustic farmhouse with a charming atmosphere",
    "Stylish loft with industrial chic decor"#,
    # ... add more descriptions ...
]

# Add the new descriptions to the dataset
data['Description'] = np.random.choice(additional_descriptions, size=len(data))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('MEDV', axis=1), data['MEDV'], test_size=0.2, random_state=42
)

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.columns.difference(['Description'])),
        ('text', CountVectorizer(), 'Description')
    ]
)

# Combine preprocessing with the model
model = make_pipeline(preprocessor, LinearRegression())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')

# Save the model for future use
joblib.dump(model, 'housing_price_model.joblib')

# Example: Predict the price for a new house
new_house = pd.DataFrame({
    'MedInc': [3.0],  # Example numerical features, use appropriate values from the dataset
    'HouseAge': [20.0],
    'AveRooms': [5.0],
    'AveBedrms': [2.0],
    'Population': [1000.0],
    'AveOccup': [3.0],
    'Latitude': [37.5],
    'Longitude': [-122.5],
    'Description': ["Charming cottage with a garden"]  # Example text feature
})

# Load the pre-trained model
loaded_model = joblib.load('housing_price_model.joblib')

# Make predictions for the new house
predicted_price = loaded_model.predict(new_house) * 100000
print(f'Predicted Price for the New House: ${predicted_price[0]:.2f}')
