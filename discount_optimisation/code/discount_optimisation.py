# Import the libraries
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)
fake = Faker()

# Define the function to generate sku data
def generate_sku():
    hexanumeric_characters = '0123456789ABCDEF'
    return ''.join(np.random.choice(list(hexanumeric_characters), size=6))



# Define the function to generate customer data
def generate_customer_data(num_customers):
  # Create an empty list to store the customer data
  customer_data = []
  
  # Loop over the number of customers
  for i in range(num_customers):
    # Generate a random age between 18 and 80
    age = np.random.randint(18, 81)
    
    # Generate a random recency between 0 and 365
    recency = np.random.randint(0, 366)
    
    # Generate a random frequency between 0 and 12
    frequency = np.random.randint(0, 13)
    
    # Generate a random value between 0 and 10000
    value = np.random.randint(0, 10001)
    
    # Generate a random last purchased order value between 0 and 1000
    last_order_value = np.random.randint(0, 1001)
    
    # Generate a random last purchased order date between 1 and 365
    last_order_date = np.random.randint(1, 366)
    
    # Generate a random city and country using faker
    city = fake.city()
    country = fake.country()
    
    # Generate a random sex using faker
    sex = fake.simple_profile()['sex']
    
    # Generate a random income range using numpy choice
    income_range = np.random.choice(['low', 'medium', 'high', 'very high', 'unknown'])
    
    # Generate a random environmental friendliness using numpy choice
    environmental_friendliness = np.random.choice(['yes', 'no'])
    
    # Generate a random hobby using faker
    hobby = fake.word()
    
    # Generate a random browsing history between 0 and 100
    browsing_history = np.random.randint(0, 101)
    
    # Generate a random social media engagement between 0 and 100
    social_media_engagement = np.random.randint(0, 101)
    
    # Generate a random personality using numpy choice
    personality = np.random.choice(['risk-averse', 'adventurous', 'impulsive', 'rational', 'unknown'])
    
    # Generate a random list of SKUs for the last 6 orders using faker
    skus = [generate_sku() for _ in range(6)]
    
    # Append the customer data to the list
    customer_data.append([age, recency, frequency, value, last_order_value, last_order_date, city, country, sex, income_range, environmental_friendliness, hobby, browsing_history, social_media_engagement, personality, skus])
  
  # Convert the list to a pandas dataframe
  customer_df = pd.DataFrame(customer_data, columns=['age', 'recency', 'frequency', 'value', 'last_order_value', 'last_order_date', 'city', 'country', 'sex', 'income_range', 'environmental_friendliness', 'hobby', 'browsing_history', 'social_media_engagement', 'personality', 'skus'])
  
  # Return the dataframe
  return customer_df



# Define the function to perform feature engineering
def feature_engineering(customer_df):
  # Create a copy of the dataframe
  customer_df = customer_df.copy()
  
  # Encode the categorical features
  # One-hot encode the city and country features
  ohe = OneHotEncoder(sparse=False)
  city_country = ohe.fit_transform(customer_df[['city', 'country']])
  city_country_cols = ohe.get_feature_names_out(['city', 'country'])
  city_country_df = pd.DataFrame(city_country, columns=city_country_cols)
  
  # Label encode the sex feature
  le = LabelEncoder()
  sex = le.fit_transform(customer_df['sex'])
  sex_col = ['sex']
  sex_df = pd.DataFrame(sex, columns=sex_col)
  
  # Ordinal encode the income range feature
  oe = OrdinalEncoder(categories=[['low', 'medium', 'high', 'very high', 'unknown']])
  income_range = oe.fit_transform(customer_df[['income_range']])
  income_range_col = ['income_range']
  income_range_df = pd.DataFrame(income_range, columns=income_range_col)
  
  # Ordinal encode the environmental friendliness feature
  oe = OrdinalEncoder(categories=[['no', 'yes']])
  environmental_friendliness = oe.fit_transform(customer_df[['environmental_friendliness']])
  environmental_friendliness_col = ['environmental_friendliness']
  environmental_friendliness_df = pd.DataFrame(environmental_friendliness, columns=environmental_friendliness_col)
  
  # One-hot encode the hobby feature
  ohe = OneHotEncoder(sparse=False)
  hobby = ohe.fit_transform(customer_df[['hobby']])
  hobby_cols = ohe.get_feature_names_out(['hobby'])
  hobby_df = pd.DataFrame(hobby, columns=hobby_cols)
  
  # One-hot encode the personality feature
  ohe = OneHotEncoder(sparse=False)
  personality = ohe.fit_transform(customer_df[['personality']])
  personality_cols = ohe.get_feature_names_out(['personality'])
  personality_df = pd.DataFrame(personality, columns=personality_cols)
  
  # Count vectorize the skus feature
  cv = CountVectorizer()
  skus = cv.fit_transform(customer_df['skus'].apply(lambda x: ' '.join(x)))
  skus_cols = cv.get_feature_names_out()
  skus_df = pd.DataFrame(skus.toarray(), columns=skus_cols)
  
  # Concatenate the encoded features with the original dataframe
  customer_df = pd.concat([customer_df, city_country_df, sex_df, income_range_df, environmental_friendliness_df, hobby_df, personality_df, skus_df], axis=1)
  
  # Drop the original categorical features
  customer_df = customer_df.drop(['city', 'country', 'sex', 'income_range', 'environmental_friendliness', 'hobby', 'personality', 'skus'], axis=1)
  
  # Scale the numerical features
  # Min-max scale the age and last order date features
  mms = MinMaxScaler()
  age_last_order_date = mms.fit_transform(customer_df[['age', 'last_order_date']])
  age_last_order_date_cols = ['age', 'last_order_date']
  age_last_order_date_df = pd.DataFrame(age_last_order_date, columns=age_last_order_date_cols)
  
  # Standard scale the value and browsing history features
  ss = StandardScaler()
  value_browsing_history = ss.fit_transform(customer_df[['value', 'browsing_history']])
  value_browsing_history_cols = ['value', 'browsing_history']
  value_browsing_history_df = pd.DataFrame(value_browsing_history, columns=value_browsing_history_cols)
  
  # Create new features
  # Create a loyalty feature as the ratio of frequency to recency
  customer_df['loyalty'] = customer_df['frequency'] / (customer_df['recency'] + 1)
  
  # Create an engagement feature as the sum of browsing history and social media engagement
  customer_df['engagement'] = customer_df['browsing_history'] + customer_df['social_media_engagement']

  # Robust scale the recency, frequency, last order value, and social media engagement features
  rs = RobustScaler()
  recency_frequency_last_order_value_social_media_engagement = rs.fit_transform(customer_df[['recency', 'frequency', 'last_order_value', 'social_media_engagement','loyalty','engagement']])
  recency_frequency_last_order_value_social_media_engagement_cols = ['recency', 'frequency', 'last_order_value', 'social_media_engagement','loyalty','engagement']
  recency_frequency_last_order_value_social_media_engagement_df = pd.DataFrame(recency_frequency_last_order_value_social_media_engagement, columns=recency_frequency_last_order_value_social_media_engagement_cols)
  
  # Concatenate the scaled features with the original dataframe
  customer_df = pd.concat([customer_df, age_last_order_date_df, value_browsing_history_df, recency_frequency_last_order_value_social_media_engagement_df], axis=1)
  
  # Drop the original numerical features
  customer_df = customer_df.drop(['age', 'recency', 'frequency', 'value', 'last_order_value', 'last_order_date', 'browsing_history', 'social_media_engagement','loyalty','engagement'], axis=1)
  
  # Return the dataframe
  return customer_df



# Define the function to train and evaluate the model
def train_and_evaluate_model(customer_df):
  # Create a copy of the dataframe
  customer_df = customer_df.copy()
  
  # Generate some random discount types for the target variable
  customer_df['discount_type'] = np.random.choice(['percentage', 'fixed', 'free shipping', 'bundled'], size=len(customer_df))
  
  # Split the dataframe into features and target
  X = customer_df.drop('discount_type', axis=1)
  y = customer_df['discount_type']
  
  # Split the data into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Define a list of models to compare
  models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), SVC()]
  model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine']
  
  # Loop over the models and compare their performance
  for i, model in enumerate(models):
    # Print the model name
    print(f'Model: {model_names[i]}')
    
    # Perform cross-validation on the train set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation score: {cv_scores.mean()}')
    
    # Fit the model on the train set
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model on the test set
    test_score = accuracy_score(y_test, y_pred)
    print(f'Test score: {test_score}')
    
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['percentage', 'fixed', 'free shipping', 'bundled'], yticklabels=['percentage', 'fixed', 'free shipping', 'bundled'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion matrix for {model_names[i]}')
    plt.show()
    
    # Print the classification report
    print(classification_report(y_test, y_pred))
    
    # Print a blank line
    print()

# Generate 1000 customer data and apply feature engineering to it
customer_df = feature_engineering(generate_customer_data(1000))

# Call the function to train and evaluate the model
train_and_evaluate_model(customer_df)
