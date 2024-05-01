import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
import torch

# Specify the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
df = pd.read_csv('/content/combined_data.csv')

# Preprocess the data
df['text'] = df['text'].apply(lambda x: x.lower())  # convert to lowercase

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Load a pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Move the model to the device
model.to(device)

# Convert texts to input IDs
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# Create a Dataset class
class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Convert our data into torch Dataset
train_dataset = EmailDataset(train_encodings, train_labels.tolist())
test_dataset = EmailDataset(test_encodings, test_labels.tolist())

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluation is performed at the end of each epoch
    save_strategy="epoch",           # save model at the end of each epoch
    fp16=True if device.type == 'cuda' else False,  # mixed precision training if GPU is available
    load_best_model_at_end=True     # Load the best model found during training at the end of training
)

# Create the Trainer and train the model
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # early stopping callback
)

trainer.train()

# Generate a spam email
spam_email = """
Subject: Exclusive Offer for You!

Dear Subscriber,

You've been selected to receive an exclusive offer! Claim your prize now by clicking the link below:

Claim Your Prize: http://spammylink.com

Hurry, this offer is only available for a limited time!

Best Regards,
Spammy Marketing Team
"""

# Tokenize the spam email
spam_email_encoding = tokenizer(spam_email, truncation=True, padding=True, return_tensors='pt')
spam_email_encoding = {k: v.to(device) for k, v in spam_email_encoding.items()}  # Move the encoding to the device

# Make predictions
logits = model(**spam_email_encoding).logits
probabilities_sp = torch.softmax(logits, dim=-1)
prediction_sp = torch.argmax(probabilities_sp)
print(f"The email is {'spam' if prediction_sp.item() == 1 else 'not spam'}")

# Generate a normal email
normal_email = """
Dear Team,

I trust this message finds you well. I wanted to bring to your attention a discrepancy in order number 12345. 
Unfortunately, one item was not included in the delivery. Given that it was offered as a complimentary item, 
adjusting the price may not resolve the issue.

Could you please advise on the best course of action to rectify this situation?

Looking forward to your prompt response.

Best Regards,
ABC
"""

# Tokenize the normal email
normal_email_encoding = tokenizer(normal_email, truncation=True, padding=True, return_tensors='pt')
normal_email_encoding = {k: v.to(device) for k, v in normal_email_encoding.items()}  # Move the encoding to the device

# Make predictions
logits = model(**normal_email_encoding).logits
probabilities_norm = torch.softmax(logits, dim=-1)
prediction_norm = torch.argmax(probabilities_norm)
print(f"The email is {'spam' if prediction_norm.item() == 1 else 'not spam'}")
