# Download the Dataset from Kaggle
# Hide your Kaggle username and password
!pip install kaggle
from google.colab import files

# Upload your kaggle.json file
uploaded = files.upload()

# Move kaggle.json to the correct directory
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset (example: 'creditcardfraud' dataset)
!kaggle datasets download -d mlg-ulb/creditcardfraud

# Unzip the dataset
!unzip creditcardfraud.zip

# Clean the Data
import pandas as pd

df = pd.read_csv('creditcard.csv')

# Checking for null values
print(df.isnull().sum())

# No null values in this dataset, so no cleaning needed

# Install Necessary Libraries
!pip install scikit-learn numpy matplotlib

# Split the Dataset into Training and Testing
from sklearn.model_selection import train_test_split
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply the Model on the Dataset
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Generate Confusion Matrix and Performance Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Generate and format the classification report
class_report_dict = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report_dict).transpose()

# Visualize confusion matrix using seaborn
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualize the classification report
plt.figure(figsize=(10, 7))
sns.heatmap(class_report_df.iloc[:-1, :].T, annot=True, cmap='Blues', cbar=False, fmt='.2f')
plt.title('Classification Report')
plt.show()

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'ROC AUC Score: {roc_auc}')

# Generate a Balanced Dataset Reflecting Type 1 and Type 2 Errors
import numpy as np

# Get counts of each class in the original dataset
class_counts = df['Class'].value_counts()
fraud_count = class_counts[1]
non_fraud_count = class_counts[0]

# Generate balanced data
fraud_data = df[df['Class'] == 1]
non_fraud_data = df[df['Class'] == 0].sample(fraud_count, random_state=42)

balanced_data = pd.concat([fraud_data, non_fraud_data])

# Introducing Type 1 and Type 2 errors
# Assume similar error rates as found in the original model's performance

# Calculate error rates
type1_error_rate = conf_matrix[0, 1] / sum(conf_matrix[0])
type2_error_rate = conf_matrix[1, 0] / sum(conf_matrix[1])

# Introduce errors in the balanced dataset
num_type1_errors = int(type1_error_rate * fraud_count)
num_type2_errors = int(type2_error_rate * fraud_count)

# Randomly flip some of the labels to introduce errors
balanced_data_errors = balanced_data.copy()
flip_indices_type1 = balanced_data_errors[balanced_data_errors['Class'] == 0].sample(num_type1_errors, random_state=42).index
flip_indices_type2 = balanced_data_errors[balanced_data_errors['Class'] == 1].sample(num_type2_errors, random_state=42).index

balanced_data_errors.loc[flip_indices_type1, 'Class'] = 1  # Introduce Type 1 errors (false positives)
balanced_data_errors.loc[flip_indices_type2, 'Class'] = 0  # Introduce Type 2 errors (false negatives)

# Split the new dataset
X_balanced = balanced_data_errors.drop('Class', axis=1)
y_balanced = balanced_data_errors['Class']

X_balanced_train, X_balanced_test, y_balanced_train, y_balanced_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Apply the Model on the Balanced Dataset
y_balanced_pred = model.predict(X_balanced_test)

# Test the model on the balanced dataset
balanced_conf_matrix = confusion_matrix(y_balanced_test, y_balanced_pred)
balanced_accuracy = accuracy_score(y_balanced_test, y_balanced_pred)
balanced_f1 = f1_score(y_balanced_test, y_balanced_pred)
balanced_roc_auc = roc_auc_score(y_balanced_test, y_balanced_pred)

# Generate and format the classification report
balanced_class_report_dict = classification_report(y_balanced_test, y_balanced_pred, output_dict=True)
balanced_class_report_df = pd.DataFrame(balanced_class_report_dict).transpose()

# Visualize confusion matrix using seaborn
plt.figure(figsize=(10,7))
sns.heatmap(balanced_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualize the classification report
plt.figure(figsize=(10, 7))
sns.heatmap(balanced_class_report_df.iloc[:-1, :].T, annot=True, cmap='Blues', cbar=False, fmt='.2f')
plt.title('Classification Report')
plt.show()

# Other KPIs
print(f'Balanced Dataset Confusion Matrix:\n{balanced_conf_matrix}')
print(f'Balanced Dataset Accuracy: {balanced_accuracy}')
print(f'Balanced Dataset F1 Score: {balanced_f1}')
print(f'Balanced Dataset ROC AUC Score: {balanced_roc_auc}')
