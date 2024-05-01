# Install the required libraries
#!pip install faker tensorflow scikit-learn pandas numpy joblib

# Import necessary libraries
import pandas as pd
from faker import Faker
import random
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Set seed for reproducibility
random.seed(42)

# Initialize Faker and create a DataFrame
fake = Faker()
num_records = 5000
df = pd.DataFrame(columns=['Age', 'Gender', 'Salary', 'Purchase_Behavior', 'Online_Channel',
                           'Last_Products_Ordered', 'Region', 'Country', 'CLV', 'NPS'])

# Populate the DataFrame with synthetic data
data_rows = []

for _ in range(num_records):
    data_rows.append({
        'Age': random.randint(18, 60),
        'Gender': random.choice(['Male', 'Female']),
        'Salary': random.randint(30000, 100000),
        'Purchase_Behavior': random.choice(['Frequent', 'Occasional', 'Rare']),
        'Online_Channel': random.choice(['Social Media', 'Search Engine', 'Direct']),
        'Last_Products_Ordered': [fake.word() for _ in range(random.randint(5, 6))],
        'Region': fake.state(),
        'Country': fake.country(),
        'CLV': round(np.random.uniform(100, 1000), 2),
        'NPS': random.randint(0, 10),
    })

df = pd.concat([df, pd.DataFrame(data_rows)], ignore_index=True)

# Save the synthetic dataset
df.to_csv('synthetic_customer_data.csv', index=False)

# Load the synthetic dataset
data = pd.read_csv('synthetic_customer_data.csv')

# Feature engineering and encoding
label_encoder = LabelEncoder()

# Initialize dictionaries to map labels to integers
gender_mapping = {}
purchase_behavior_mapping = {}
online_channel_mapping = {}

# Add a new category for unseen labels and fit_transform
data['Gender'] = data['Gender'].astype('str').apply(lambda x: gender_mapping.setdefault(x, len(gender_mapping)))
data['Purchase_Behavior'] = data['Purchase_Behavior'].astype('str').apply(lambda x: purchase_behavior_mapping.setdefault(x, len(purchase_behavior_mapping)))
data['Online_Channel'] = data['Online_Channel'].astype('str').apply(lambda x: online_channel_mapping.setdefault(x, len(online_channel_mapping)))

features = data[['Age', 'Salary', 'Gender', 'Purchase_Behavior', 'Online_Channel', 'CLV', 'NPS']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# KMeans clustering with explicit n_init
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data['Segment'] = kmeans.fit_predict(scaled_features)

# Save the KMeans model using joblib
kmeans_model_path = 'customer_segmentation_model.joblib'
joblib.dump(kmeans, kmeans_model_path)

# Evaluate the model using silhouette score
silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

# Generate a new list of fake customers
new_customers_data = []

for _ in range(num_records):
    new_customer = {
        'Age': random.randint(18, 60),
        'Gender': random.choice(['Male', 'Female']),
        'Salary': random.randint(30000, 100000),
        'Purchase_Behavior': random.choice(['Frequent', 'Occasional', 'Rare']),
        'Online_Channel': random.choice(['Social Media', 'Search Engine', 'Direct']),
        'CLV': round(np.random.uniform(100, 1000), 2),
        'NPS': random.randint(0, 10),
    }
    new_customers_data.append(new_customer)

# Create a DataFrame from the list of dictionaries
new_customers = pd.DataFrame(new_customers_data, columns=features.columns)

# Encode the new customer data
new_customers['Gender'] = new_customers['Gender'].astype('str').apply(lambda x: gender_mapping.setdefault(x, len(gender_mapping)))
new_customers['Purchase_Behavior'] = new_customers['Purchase_Behavior'].astype('str').apply(lambda x: purchase_behavior_mapping.setdefault(x, len(purchase_behavior_mapping)))
new_customers['Online_Channel'] = new_customers['Online_Channel'].astype('str').apply(lambda x: online_channel_mapping.setdefault(x, len(online_channel_mapping)))

# Scale the new customer data
scaled_new_customers = scaler.transform(new_customers)

# Load the saved model using joblib
loaded_model = joblib.load(kmeans_model_path)

# Predict segments for new customers
new_customers['Segment'] = loaded_model.predict(scaled_new_customers)

# Display the results
print(new_customers)
