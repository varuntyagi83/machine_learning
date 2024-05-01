# Install necessary libraries
#!pip install transformers scikit-learn torch

# Import necessary libraries
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

# Download and load the "German Credit" dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = [
    "existing_checking", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status", "other_parties",
    "residence_since", "property_magnitude", "cc_age", "other_payment_plans",
    "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker",
    "class"
]
df = pd.read_csv(url, sep=' ', names=column_names, na_values=["?"])

# Drop rows with missing values
df = df.dropna()

# Preprocess the data
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])  # Convert labels to numerical values
X = df.drop('class', axis=1)
y = df['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the input data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train_tokens = tokenizer(list(X_train['purpose']), padding=True, truncation=True, return_tensors="pt", max_length=128)
X_test_tokens = tokenizer(list(X_test['purpose']), padding=True, truncation=True, return_tensors="pt", max_length=128)

# Prepare DataLoader for training and testing
train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], torch.tensor(y_train.values))
test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], torch.tensor(y_test.values))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Train the model
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Test the model
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on the test set: {accuracy}")

# Predict for a new person
def predict_loan_approval(description):
    model.eval()
    description_str = " ".join(map(str, description))  # Convert the input list to a string
    inputs = tokenizer(description_str, padding=True, truncation=True, return_tensors="pt", max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return le.inverse_transform([prediction])[0]

# Example usage for a new person
new_person_description = ["<no checking>", "<15>", "existing paid", "radio/tv", "<5000>",
                          "<500>", "<1>", "<2>", "male single", "none",
                          "<1>", "real estate", "<25>", "none", "own",
                          "<1>", "skilled", "<1>", "yes, registered under the customer's name", "no"]
predicted_class = predict_loan_approval(new_person_description)

# Print whether the loan is approved or not
if predicted_class == 1:
    print("Congratulations! The loan is approved.")
else:
    print("Sorry, the loan is not approved.")
