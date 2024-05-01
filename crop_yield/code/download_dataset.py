import os
import zipfile
from zipfile import ZipFile
from google.colab import files

# Upload your Kaggle API key (kaggle.json) using the Colab interface
files.upload()

# Move the uploaded key to the correct location
!mkdir ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

from kaggle.api.kaggle_api_extended import KaggleApi

# Instantiate the Kaggle API client
api = KaggleApi()
api.authenticate()

# Define the directory path where you want to download the dataset
download_dir = "/content"

# Download the dataset into the specified directory
api.dataset_download_files(dataset="patelris/crop-yield-prediction-dataset", path=download_dir, unzip=True)
