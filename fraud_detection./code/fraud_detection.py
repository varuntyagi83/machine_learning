# Importing the necessary libraries
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ... (rest of the code)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature names
feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train IsolationForest model
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train_scaled)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Convert predictions to binary labels (1 for anomalies, 0 for normal)
binary_predictions = np.where(predictions == -1, 1, 0)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, binary_predictions)
classification_rep = classification_report(y_test, binary_predictions, zero_division=1)

# Display the results
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
