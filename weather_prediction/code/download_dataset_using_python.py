import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta

base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data?"
token = "YOUR_NCEI_API_TOKEN"  # Replace with your valid token

dataset = "GHCND"  # Adjust as needed (e.g., GSNOD, GSOM)
station_id = "USW00094297"  # Replace with your desired station ID
start_date = "2024-02-10"  # Adjust as needed (YYYY-MM-DD)
end_date = datetime.today().strftime("%Y-%m-%d")  # Dynamic end date
variables = "TMAX,TMIN,PRCP"  # Adjust as needed (e.g., TAVG, RH, WIND)

# Replace the dataset and station ID with your specific requirements
params = {"datasetid": dataset, "stationid": station_id,
         "startdate": start_date, "enddate": end_date,
         "variables": variables, "token": token}

# The code incorporates basic error handling. 
# Consider more robust strategies in production settings.
try:
    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise an exception for non-200 status codes
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    exit()

# Extract data from response
data = response.text

df = pd.read_csv(StringIO(data), skipinitialspace=True, delimiter=",")

# Convert dates and handle missing values
df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
df = df.dropna(subset=["DATE", "TMAX", "TMIN", "PRCP"])

# Access forecast data (might vary depending on the dataset)
forecasts = df[df["DATETYPE"] == "FORECAST"]  # Adjust the "DATETYPE" column name
print(forecasts)
