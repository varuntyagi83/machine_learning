# Install necessary libraries
# pip install numpy pandas tensorflow transformers requests zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import requests
from zipfile import ZipFile
from io import BytesIO

# Download and extract MovieLens dataset
url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
response = requests.get(url)
with ZipFile(BytesIO(response.content)) as zip_file:
    with zip_file.open('ml-latest-small/movies.csv') as movies_file:
        movies_data = pd.read_csv(movies_file)
    with zip_file.open('ml-latest-small/ratings.csv') as ratings_file:
        ratings_data = pd.read_csv(ratings_file)

# Merge 'movies.csv' and 'ratings.csv' on the 'movieId' column
movielens_data = pd.merge(ratings_data, movies_data, on='movieId')

# Keep only relevant columns (you may adjust this based on your specific requirements)
movielens_data = movielens_data[['movieId', 'title', 'genres', 'rating']]

# Use a subset of the data for faster training
subset_size = 1000
movielens_data_subset = movielens_data.sample(subset_size, random_state=42)

# Load pre-trained DistilBERT model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize and train the model
max_tokenization_length = 128  # Adjust as needed
inputs = tokenizer(movielens_data_subset['genres'].astype(str).tolist(), return_tensors='tf', truncation=True, padding=True, max_length=max_tokenization_length)
labels = movielens_data_subset['rating']

# Convert tokenized inputs to NumPy arrays
inputs = {key: np.array(value) for key, value in inputs.items()}

# Adjust hyperparameters based on your use case
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
model.fit(inputs, labels, epochs=5, batch_size=32)  # Adjust batch size as needed



# Generate synthetic data for recommendations
synthetic_data = pd.DataFrame({
    'title': movielens_data_subset['title'].head(2).tolist(),
    'genres': movielens_data_subset['genres'].head(2).tolist()
})

# Tokenize synthetic data and get predictions
synthetic_inputs = tokenizer(synthetic_data['genres'].astype(str).tolist(), return_tensors='tf', truncation=True, padding=True, max_length=max_tokenization_length)
synthetic_inputs = {key: np.array(value) for key, value in synthetic_inputs.items()}
synthetic_predictions = model.predict(synthetic_inputs)['logits']

# Check and handle length mismatch
title_length = len(synthetic_data['title'])
predictions_length = len(synthetic_predictions.flatten())

if title_length != predictions_length:
    print(f"WARNING: Length mismatch between titles ({title_length}) and predictions ({predictions_length}).")

    # Choose the minimum length for safe handling
    min_length = min(title_length, predictions_length)

    # Adjust predictions length (either truncate or pad)
    if predictions_length > min_length:
        synthetic_predictions = synthetic_predictions.flatten()[:min_length]
    else:
        # Implement padding if needed (using np.pad or similar)
        pass

# Create and display recommendations
recommendations = pd.DataFrame({
    'Movie': synthetic_data['title'].tolist()[:min_length], # Use adjusted title length
    'Predicted Rating': synthetic_predictions
})

print(recommendations)
