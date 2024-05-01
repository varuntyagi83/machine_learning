# Import necessary libraries
import numpy as np
import pandas as pd
import random
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
import string

# Set seed for reproducibility
np.random.seed(0)
random.seed(0)

# Function to generate 12-digit alphanumeric lead ids
def generate_lead_id():
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(12))

# Generate fake lead data
fake = Faker()
num_samples = 1000

lead_data = []
for _ in range(num_samples):
    lead_data.append([
        generate_lead_id(),  # 12-digit alphanumeric lead id
        fake.date_between(start_date='-30d', end_date='today'),  # Date of the lead
        fake.date_between(start_date='today', end_date='+30d'),  # Date of moving
        fake.random_int(min=400, max=2000),  # Surface area of the apartment (in square feet)
        fake.random_int(min=5, max=100),  # Volume of the furniture (in cubic feet)
        fake.random_element(elements=('North', 'South', 'East', 'West')),  # Region of the lead
        fake.random_element(elements=('Germany', 'France', 'Sweden')),  # Country
        fake.random_element(elements=('Social Media', 'Referrals', 'Website', 'Phone', 
                                      'Immoscout24', 'Immowelt', 'Demenagement', 'Ebay', 
                                      'Wunderflats', 'Wg-gesucht', 'HousingAnywhere')),  # Source of the lead
        fake.random_element(elements=('Low', 'Medium', 'High')),  # Urgency of the move
        fake.random_element(elements=(0, 1))  # Binary conversion rate (0 or 1)
    ])

# Convert lead_data to DataFrame
lead_df = pd.DataFrame(lead_data, columns=[
    'Lead ID', 'Date of the lead', 'Date of moving',
    'Surface area of the apartment', 'Volume of the furniture',
    'Region of the lead', 'Country', 'Source of the lead', 'Urgency of the move', 'Conversion Rate'
])

# Convert date columns to datetime objects
lead_df['Date of the lead'] = pd.to_datetime(lead_df['Date of the lead'])
lead_df['Date of moving'] = pd.to_datetime(lead_df['Date of moving'])

# Extract relevant information from the date features
lead_df['Day of the week'] = lead_df['Date of the lead'].dt.dayofweek
lead_df['Days until moving'] = (lead_df['Date of moving'] - lead_df['Date of the lead']).dt.days

# Drop the original date features and 'Lead ID'
lead_df = lead_df.drop(['Date of the lead', 'Date of moving', 'Lead ID'], axis=1)

# Convert categorical features to one-hot encoding
lead_df_encoded = pd.get_dummies(lead_df, columns=['Region of the lead', 'Country', 'Source of the lead', 'Urgency of the move'])

# Separate features (X) and target variable (y)
X = lead_df_encoded.drop('Conversion Rate', axis=1).values
y = lead_df_encoded['Conversion Rate'].astype(int).values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Resample the training data to handle class imbalance
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)  # Use only numeric columns
X_test_scaled = scaler.transform(X_test)

# Build a Neural Network model using Keras without class weights parameter
model = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=1000, random_state=42, warm_start=True)

# Train the model
model.fit(X_train_resampled_scaled, y_train_resampled)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print accuracy and confusion matrix for each country
countries = lead_df['Country'].unique()
for country in countries:
    country_mask = lead_df['Country'] == country
    X_country = lead_df_encoded[country_mask].drop('Conversion Rate', axis=1).values
    y_country = lead_df_encoded[country_mask]['Conversion Rate'].astype(int).values
    
    X_country_scaled = scaler.transform(X_country)
    predictions_country = model.predict(X_country_scaled)
    
    accuracy_country = accuracy_score(y_country, predictions_country)
    print(f"Accuracy for {country}: {accuracy_country * 100:.2f}%")
    
    # Print confusion matrix
    cm = confusion_matrix(y_country, predictions_country)
    labels = unique_labels(y_country, predictions_country)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {country}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Generate another set of fake lead data for prediction
fake_lead_data = []
for _ in range(100):
    fake_lead_data.append([
        generate_lead_id(),  # 12-digit alphanumeric lead id
        fake.date_between(start_date='-30d', end_date='today'),  # Date of the lead
        fake.date_between(start_date='today', end_date='+30d'),  # Date of moving
        fake.random_int(min=400, max=2000),  # Surface area of the apartment (in square feet)
        fake.random_int(min=5, max=100),  # Volume of the furniture (in cubic feet)
        fake.random_element(elements=('North', 'South', 'East', 'West')),  # Region of the lead
        fake.random_element(elements=('Germany', 'France', 'Sweden')),  # Country
        fake.random_element(elements=('Social Media', 'Referrals', 'Website', 'Phone', 
                                      'Immoscout24', 'Immowelt', 'Demenagement', 'Ebay', 
                                      'Wunderflats', 'Wg-gesucht', 'HousingAnywhere')),  # Source of the lead
        fake.random_element(elements=('Low', 'Medium', 'High')),  # Urgency of the move
    ])

# Convert fake_lead_data to DataFrame
fake_lead_df = pd.DataFrame(fake_lead_data, columns=[
    'Lead ID', 'Date of the lead', 'Date of moving',
    'Surface area of the apartment', 'Volume of the furniture',
    'Region of the lead', 'Country', 'Source of the lead', 'Urgency of the move'
])

# Convert date columns to datetime objects
fake_lead_df['Date of the lead'] = pd.to_datetime(fake_lead_df['Date of the lead'])
fake_lead_df['Date of moving'] = pd.to_datetime(fake_lead_df['Date of moving'])

# Extract relevant information from the date features
fake_lead_df['Day of the week'] = fake_lead_df['Date of the lead'].dt.dayofweek
fake_lead_df['Days until moving'] = (fake_lead_df['Date of moving'] - fake_lead_df['Date of the lead']).dt.days

# Drop the original date features and 'Lead ID'
fake_lead_df = fake_lead_df.drop(['Date of moving', 'Lead ID'], axis=1)

# Convert categorical features to one-hot encoding
fake_lead_df_encoded = pd.get_dummies(fake_lead_df, columns=['Region of the lead', 'Country', 'Source of the lead', 'Urgency of the move'])

# Drop the 'Date of the lead' column before scaling
fake_lead_df_encoded = fake_lead_df_encoded.drop(['Date of the lead'], axis=1)

# Standardize features by removing the mean and scaling to unit variance
fake_lead_X_scaled = scaler.transform(fake_lead_df_encoded.values)

# Make predictions on the new set of fake leads with probability estimates
fake_lead_probabilities = model.predict_proba(fake_lead_X_scaled)[:, 1]  # Probability of conversion

# Display predictions along with country and lead creation date
fake_lead_results = pd.DataFrame({
    'Lead ID': [lead[0] for lead in fake_lead_data],  # Extract 'Lead ID' from fake_lead_data
    'Country': fake_lead_df['Country'],
    'Lead Creation Date': fake_lead_df['Date of the lead'],
    'Probability of Conversion': fake_lead_probabilities
})

# Sort the results DataFrame by "Probability of Conversion" in descending order for each country
fake_lead_results_sorted = fake_lead_results.sort_values(by=['Country', 'Probability of Conversion'], ascending=[True, False])

print(fake_lead_results_sorted)
