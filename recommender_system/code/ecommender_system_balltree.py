import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Generate random data for customers
customer_data = {
    'user_id': np.arange(1, 501),
    'age': np.random.randint(18, 60, size=500),
    'gender': np.random.choice(['M', 'F'], size=500),
    'location': np.random.choice(['City', 'Suburb', 'Rural'], size=500),
    'clicks': np.random.randint(5, 30, size=500),
    'browsing_behavior': np.random.randint(10, 40, size=500),
    'items_in_cart': np.random.randint(0, 5, size=500),
}

customers_df = pd.DataFrame(customer_data)

# Generate random data for items
item_data = {
    'product_id': np.arange(101, 1601),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], size=1500),
    'brand': np.random.choice(['Brand1', 'Brand2', 'Brand3'], size=1500),
    'price': np.random.uniform(10, 200, size=1500),
    'predicted_sales': np.random.randint(1, 10, size=1500),
    'marketing_campaign': np.random.choice(['Campaign1', 'Campaign2', 'Campaign3'], size=1500),
    'bundle': np.random.choice([0, 1], size=1500),
}

items_df = pd.DataFrame(item_data)

# Merge customer and item data
df = pd.merge(customers_df, items_df, how='cross')

# Data Preprocessing
df_encoded = pd.get_dummies(df, columns=['gender', 'location', 'category', 'brand', 'marketing_campaign'])

# Feature Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded[['age', 'clicks', 'browsing_behavior', 'items_in_cart', 'price']])

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Combine features
df_combined = pd.concat([df_encoded, pd.DataFrame(df_pca, columns=['pca1', 'pca2'])], axis=1)

# Train-Test Split
train_data, test_data = train_test_split(df_combined, test_size=0.2, random_state=42)

# Model Building (BallTree)
model = BallTree(train_data[['predicted_sales', 'pca1', 'pca2']], metric='minkowski')

# Model Prediction
_, indices = model.query(test_data[['predicted_sales', 'pca1', 'pca2']], k=5)

# Recommendations
recommendations = []
for _, row in test_data.iterrows():
    neighbor_products = train_data.iloc[indices[0]]['product_id'].explode().tolist()
    recommended_products = list(set(neighbor_products))
    recommendations.append({'user_id': row['user_id'], 'recommended_products': recommended_products})

# Evaluate the Model
predictions = []
for _, row in test_data.iterrows():
    neighbor_products = train_data.iloc[indices[0]]['predicted_sales'].explode().tolist()
    predicted_sales = np.mean(neighbor_products)
    predictions.append({'user_id': row['user_id'], 'predicted_sales': predicted_sales})

# Calculate Mean Squared Error
actual_sales = test_data['predicted_sales'].tolist()
predicted_sales = [pred['predicted_sales'] for pred in predictions]
mse = mean_squared_error(actual_sales, predicted_sales)
print(f"Mean Squared Error: {mse}")

# Display Recommendations
for recommendation in recommendations:
    print(f"User {recommendation['user_id']} Recommendations: {recommendation['recommended_products']}")
