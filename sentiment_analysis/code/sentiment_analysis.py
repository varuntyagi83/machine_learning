import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate synthetic customer data with comments in multiple languages and language column
def generate_synthetic_data(num_customers=1000):
    fake_en = Faker('en_US')
    fake_de = Faker('de_DE')
    fake_fr = Faker('fr_FR')
    fake_it = Faker('it_IT')

    data = {
        'Age': [],
        'Recency': [],
        'Frequency': [],
        'Order_Value': [],
        'Last_Purchase_Value': [],
        'Last_Purchase_Date': [],
        'City': [],
        'Country': [],
        'Sex': [],
        'Income_Range': [],
        'Hobbies': [],
        'NPS_Survey_Date': [],
        'NPS_Score': [],
        'NPS_Comments': [],
        'Language': []
    }

    for _ in range(num_customers):
        # Generate a random language for the customer
        language = random.choice(['en_US', 'de_DE', 'fr_FR', 'it_IT'])

        # Generate data based on the selected language
        if language == 'en_US':
            fake = fake_en
        elif language == 'de_DE':
            fake = fake_de
        elif language == 'fr_FR':
            fake = fake_fr
        elif language == 'it_IT':
            fake = fake_it

        data['Age'].append(fake.random_int(min=18, max=65))
        data['Recency'].append(fake.random_int(min=1, max=365))
        data['Frequency'].append(fake.random_int(min=1, max=10))
        data['Order_Value'].append(fake.random_int(min=20, max=200))
        data['Last_Purchase_Value'].append(fake.random_int(min=20, max=200))
        data['Last_Purchase_Date'].append(fake.date_between(start_date='-365d', end_date='today'))
        data['City'].append(fake.city())
        data['Country'].append(fake.country())
        data['Sex'].append(fake.random_element(elements=('Male', 'Female')))
        data['Income_Range'].append(fake.random_element(elements=('Low', 'Medium', 'High')))
        data['Hobbies'].append(fake.random_element(elements=('Reading', 'Traveling', 'Sports', 'Cooking')))
        data['NPS_Survey_Date'].append(fake.date_between(start_date='-365d', end_date='today'))
        data['NPS_Score'].append(fake.random_int(min=0, max=10))
        data['NPS_Comments'].append(fake.text())
        data['Language'].append(language)

    df = pd.DataFrame(data)
    return df

customer_data = generate_synthetic_data()

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Function to perform sentiment analysis on comments
def analyze_sentiment(comment):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(comment)

    # Categorize sentiment
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis to NPS comments
customer_data['Sentiment'] = customer_data['NPS_Comments'].apply(analyze_sentiment)

# Function to categorize comments based on specific aspects
def categorize_comments(comment):
    # Your logic for categorization goes here
    # Example: Categorize based on keywords like 'price', 'delivery', 'quality', etc.
    if 'price' in comment.lower():
        return 'Price Sensitivity'
    elif 'delivery' in comment.lower():
        return 'Delivery Sensitivity'
    elif 'quality' in comment.lower():
        return 'Quality Sensitivity'
    else:
        return 'Other'

# Apply categorization to NPS comments
customer_data['Comment_Category'] = customer_data['NPS_Comments'].apply(categorize_comments)

# Calculate average NPS score
average_nps_score = customer_data['NPS_Score'].mean()

# Calculate average age of customers
average_age = customer_data['Age'].mean()

# Calculate most common hobbies
most_common_hobbies = customer_data['Hobbies'].mode().tolist()

# Calculate percentage of positive, negative, and neutral sentiments
sentiment_distribution = customer_data['Sentiment'].value_counts(normalize=True) * 100

# Display insights
print(f"Average NPS Score: {average_nps_score:.2f}")
print(f"Average Age of Customers: {average_age:.2f} years")
print(f"Most Common Hobbies: {', '.join(most_common_hobbies)}")
print("Sentiment Distribution:")
print(sentiment_distribution)

# Function to print sentiment distribution by language
def print_sentiment_distribution_by_language(language):
    filtered_data = customer_data[customer_data['Language'] == language]
    sentiment_distribution = filtered_data['Sentiment'].value_counts(normalize=True) * 100

    print(f"Sentiment Distribution for {language}:")
    print(sentiment_distribution)
    print("\n")

# Assuming you have already performed sentiment analysis and categorization
# You can use the previously defined functions for sentiment analysis and categorization here

# For example:
customer_data['Sentiment'] = customer_data['NPS_Comments'].apply(analyze_sentiment)
customer_data['Comment_Category'] = customer_data['NPS_Comments'].apply(categorize_comments)

# Print sentiment distribution for each known language
for language in customer_data['Language'].unique():
    print_sentiment_distribution_by_language(language)
