import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import torch
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import BertTokenizer, BertForSequenceClassification
custom_prompt = ['ваша фраза для распознавани интента']
model_name = "cointegrated/rubert-tiny2"
tokenizer = BertTokenizer.from_pretrained(model_name)

custom_tokenized = tokenizer(custom_prompt, padding=True, truncation=True, max_length=512)
model_path = "intent_model"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3) # len(mapping) = 3
# Create torch dataset
custom_tokenized_dataset = Dataset(custom_tokenized)

# Make prediction
test_trainer = Trainer(model)
raw_pred, _, _ = test_trainer.predict(custom_tokenized_dataset)

# Preprocess raw predictions
y_pred = torch.argmax(torch.softmax(torch.tensor(raw_pred), dim=-1), axis=1)
print(y_pred) # class