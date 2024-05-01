import os
import zipfile
import urllib.request
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
import re

# Define download path (adjust as needed)
download_path = "/Users/Varun/Downloads/python_practice/"

# MovieLens dataset URL
dataset_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Check if download path exists
if not os.path.exists(download_path):
    os.makedirs(download_path)
    print("Created download directory at:", download_path)

# Check if dataset file exists
dataset_file = os.path.join(download_path, "ml-latest-small.zip")
if not os.path.exists(dataset_file):
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_file)
    print("Dataset downloaded successfully!")
else:
    print("Dataset already exists, skipping download.")

# Check if the unzipped dataset folder already exists
unzipped_folder = os.path.join(download_path, "ml-latest-small")
if not os.path.exists(unzipped_folder):
    # Extract the ZIP file if the folder doesn't exist
    with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
        zip_ref.extractall(download_path)
else:
    print("Unzipped dataset folder already exists, using existing files.")

# Load data from the unzipped folder
ratings = pd.read_csv(os.path.join(unzipped_folder, "ratings.csv"))
movies = pd.read_csv(os.path.join(unzipped_folder, "movies.csv"))
tags = pd.read_csv(os.path.join(unzipped_folder, "tags.csv"))
links = pd.read_csv(os.path.join(unzipped_folder, "links.csv"))

# Combine user & movie data
data = pd.merge(ratings, movies, on="movieId")

# Define function to prepare user data
def prepare_user_data(user_id):
    user_data = data[data["userId"] == user_id]
    
    # Get movie IDs instead of titles
    user_movies = user_data["movieId"].to_list()[:5]
    user_genres = user_data["genres"].iloc[0].split("|") if not user_data.empty else []
    user_tags = tags[tags["userId"] == user_id]["tag"].to_list() if user_id in tags["userId"].unique() else []
    return user_movies, user_genres, user_tags

# Example: Get data for User ID 1
user_id = 2
user_movies, user_genres, user_tags = prepare_user_data(user_id)

# Initialize T5Tokenizer with model_max_length
tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=1024)

# Load T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Access external movie information using links.csv and TMDB API
def get_movie_info(movie_id):
    try:
        # Check if movie ID exists in links DataFrame
        if not links.query(f"movieId == {movie_id}").empty:
            tmdb_id = links[links["movieId"] == movie_id]["tmdbId"].iloc[0]
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=d5fea57cced3ac269b1f838683f2d574"  # Replace with your TMDB API key
            response = requests.get(url)
            data = response.json()
            title = data.get("title", "Title not found")
            return title
        else:
            print(f"Movie ID {movie_id} not found in links DataFrame.")
            return "Title not found"
    except IndexError:
        print(f"Missing TMDB ID for movie ID {movie_id}.")
        return "Title not found"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie information from TMDB API: {e}")
        return "Title not found"

# Create T5 prompt with user features and movie information
prompt_user_movies = f"A user has watched {', '.join([str(movie_id) for movie_id in user_movies])}."
prompt_genres_tags = f"They seem to enjoy {', '.join(user_genres)} genres and movies with tags like {', '.join(user_tags)}."

# Combine the prompts with a delimiter
prompt = f"{prompt_user_movies} {prompt_genres_tags}"

# Use the "recommend" keyword in the prompt
prompt += " Recommend some movies they might enjoy."

# Encode prompt for model
user_input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate movie recommendations
output = model.generate(
    user_input_ids, max_length=256, num_beams=5, num_return_sequences=3
)

# Decode and print recommendations
top_recommendations = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

# Extract IDs using regular expression
pattern = r"\d+"
recommended_ids_set = set()

for i, row in enumerate(top_recommendations):
    matches = re.findall(pattern, row)
    recommended_ids_set.update(matches)
    # print(f"Recommendation {i+1}: {row}")

# Convert the set to a list if needed
recommended_ids = list(recommended_ids_set)

# Get movie titles from retrieved IDs
recommended_titles = [get_movie_info(int(recommended_id)) for recommended_id in recommended_ids]

# Print recommendations with titles
print(f"\nTop Recommendations with Titles for the user {user_id}:")
for i, title in enumerate(recommended_titles):
    print(f"{i + 1}. {title}")
