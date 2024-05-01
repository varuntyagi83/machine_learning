import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Placeholder metrics (to be replaced with actual calculations)
accuracy_threshold = 0.85
completeness_threshold = 0.8
consistency_threshold = 0.85
timeliness_threshold = 12  # hours


# Generating synthetic data
num_shipments = 1000

# Set start and end date for the random timestamps
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 2, 1)

# Calculate the difference in seconds between start and end dates
time_delta = end_date - start_date
total_seconds = time_delta.total_seconds()

# Generate 100 random timestamps within the time range
random_timestamps = np.random.randint(0, int(total_seconds), size=num_shipments)

# Convert the timestamps from seconds to datetime objects
# Ensure conversion to Python int for timedelta constructor
timestamps = start_date + np.array([timedelta(seconds=int(t)) for t in random_timestamps])

# Set some timeliness values to be true (within the threshold)
#num_true_timeliness = int(0.1 * num_shipments)
#true_timeliness_indices = np.random.choice(num_shipments, size=num_true_timeliness, replace=False)
#timestamps[true_timeliness_indices] = datetime.now() - timedelta(hours=timeliness_threshold - 1)

data = {
    'Accuracy': np.random.uniform(low=0.8, high=1, size=num_shipments),
    'Timeliness': timestamps,
    'Completeness': np.random.uniform(low=0.9, high=1, size=num_shipments),
    'Consistency': np.random.uniform(low=0.85, high=1, size=num_shipments)
}

df_shipments = pd.DataFrame(data)

# Data quality checks
def check_accuracy(data, accuracy_threshold):
    return data > accuracy_threshold


def check_completeness(data, completeness_threshold):
    return data > completeness_threshold


def check_consistency(data, consistency_threshold):
    return data / consistency_threshold


def check_timeliness(data, timeliness_threshold):
    current_time = pd.to_datetime(datetime.now())
    threshold_timeliness = current_time - timedelta(hours=timeliness_threshold)
    data_datetime = pd.to_datetime(data)  # Convert Series to datetime object
    time_difference = current_time - data_datetime
    total_seconds = time_difference.dt.total_seconds()  # Use dt accessor for Series
    threshold_seconds = timeliness_threshold * 3600  # Convert threshold to seconds
    return total_seconds <= threshold_seconds

# Check data quality
print("\nData Quality Checks:")
print("Accuracy Check:", check_accuracy(df_shipments['Accuracy'], accuracy_threshold))
print("Completeness Check:", check_completeness(df_shipments['Completeness'], completeness_threshold))
print("Consistency Check:", check_consistency(df_shipments['Consistency'], consistency_threshold))
print("Timeliness Check:", check_timeliness(df_shipments['Timeliness'], timeliness_threshold))
print(df_shipments['Timeliness'].mean())

# Plotting Radar Chart
labels = ['Accuracy', 'Completeness', 'Consistency', 'Timeliness']
values = [
    df_shipments['Accuracy'].mean(),
    df_shipments['Completeness'].mean(),
    df_shipments['Consistency'].mean(),
    (datetime.now() - df_shipments['Timeliness']).mean().total_seconds() / 3600
]

# Normalize timeliness value to fit within 0 to 1
timeliness_max = (datetime.now() - df_shipments['Timeliness'].min()).total_seconds() / 3600
values[3] = min(values[3] / timeliness_max, 1.0)

# Calculate angles for each label
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# Close the plot by repeating the first angle
angles += angles[:1]
values += values[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values, color='blue', linewidth=1)
ax.fill(angles, values, color='blue', alpha=0.25)

# Set the labels and ticks
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Set the y-axis limit
ax.set_ylim(0, 1)

# Set the title
ax.set_title('Data Quality Radar Chart')

# Display the plot
plt.show()
