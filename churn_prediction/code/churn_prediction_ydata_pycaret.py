# Step 1: Install Required Packages
pip install ydata-profiling pycaret nltk

# Step 2: Import necessary libraries
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from pycaret.classification import *
import matplotlib.pyplot as plt
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Step 3: Download NLTK sentiment lexicon
nltk.download('vader_lexicon')

# Step 4: Generate Synthetic Data
def generate_synthetic_churn_data(n_users=1000):
    np.random.seed(42)
    random.seed(42)
    
    user_ids = [f"user_{i}" for i in range(n_users)]
    interactions = np.random.poisson(5, n_users)
    time_spent = np.random.normal(loc=50, scale=10, size=n_users)
    number_of_sessions = np.random.poisson(3, n_users)
    days_since_last_login = np.random.randint(0, 30, n_users)
    avg_interactions_per_session = interactions / np.maximum(1, number_of_sessions)
    feedback = [
        random.choice(["positive", "neutral", "negative"]) for _ in range(n_users)
    ]
    churned = np.random.choice(["yes", "no"], size=n_users, p=[0.2, 0.8])

    synthetic_data = pd.DataFrame({
        "user_id": user_ids,
        "interactions": interactions,
        "time_spent": time_spent,
        "number_of_sessions": number_of_sessions,
        "days_since_last_login": days_since_last_login,
        "avg_interactions_per_session": avg_interactions_per_session,
        "feedback": feedback,
        "churned": churned
    })
    return synthetic_data

# Generate synthetic data
synthetic_data = generate_synthetic_churn_data()

# Step 5: Sentiment Analysis on Feedback within synthetic_data
sid = SentimentIntensityAnalyzer()
synthetic_data["sentiment"] = synthetic_data["feedback"].apply(lambda x: sid.polarity_scores(x)["compound"])

# Step 6: Encode Non-Numeric Columns in a New DataFrame for Model Training
# Dropping 'user_id' as it's an identifier, not useful for prediction
synthetic_data_encoded = synthetic_data.drop(columns=['user_id'])
synthetic_data_encoded = pd.get_dummies(synthetic_data_encoded, columns=["feedback"], drop_first=True)

# Step 7: Generate and Save EDA Report as HTML using ydata-profiling
profile = ProfileReport(synthetic_data_encoded, title="Synthetic Data EDA Report", explorative=True)
profile.to_file("synthetic_data_eda_report.html")
print("EDA report has been generated and saved as 'synthetic_data_eda_report.html'.")

# Step 8: PyCaret Setup for Churn Prediction with Encoded Data
synthetic_data_encoded["churned"] = synthetic_data_encoded["churned"].astype(str)

# Set up the PyCaret classification environment
clf = setup(data=synthetic_data_encoded, target="churned", session_id=42, normalize=True, 
            transformation=True, remove_outliers=True, fix_imbalance=True, html=False, verbose=False, use_gpu=False)

# Step 9: Let PyCaret Choose the Best Model
best_model = compare_models(sort="F1")  # Sort by F1-score or any other metric like accuracy

# Step 10: Model Evaluation - Confusion Matrix and Other Metrics 
evaluate_model(best_model)

# Step 11: Generate Personalized Re-Engagement Messages for Churned Users
def create_reengagement_message(feedback, sentiment_score, interactions, days_since_last_login):
    if sentiment_score < -0.5:
        if "support" in feedback.lower():
            return "We're here to help! Reach out to our support team for an improved experience."
        elif "ads" in feedback.lower():
            return "Enjoy an ad-free experience with our premium plan. Come back and explore!"
        elif "rewards" in feedback.lower():
            return "We've boosted our rewards! Log back in and start earning more today."
        else:
            return "We value your feedback! Check out the latest improvements in the app."
    elif sentiment_score > 0.5:
        return "We miss you! Come back and continue where you left off with the features you love."
    elif interactions < 3 and days_since_last_login > 7:
        return "It looks like you're missing out! Log in now to claim a special bonus."
    else:
        return "We're constantly improving. Come back and give us another try!"

# Apply the re-engagement message function to churned users in synthetic_data
synthetic_data["reengagement_message"] = synthetic_data.apply(
    lambda x: create_reengagement_message(
        x["feedback"], x["sentiment"], x["interactions"], x["days_since_last_login"]
    ) if x["churned"] == "yes" else None,
    axis=1
)

# Display a few churned users with their re-engagement messages
churned_users = synthetic_data[synthetic_data['churned'] == "yes"][['feedback', 'sentiment', 'reengagement_message']]
print(churned_users.head(10))
