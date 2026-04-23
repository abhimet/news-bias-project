import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from clearml import Task

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["bert", "distilbert"], required=True)
args = parser.parse_args()

if args.model == "bert":
    MODEL_NAME = "bert-base-uncased"
else:
    MODEL_NAME = "distilbert-base-uncased"

#all configurations
THRESHOLD = 0.80
MAX_LEN = 256
BATCH = 16
EPOCHS = 3
LR = 2e-5
DATA_PATH = "bias_clean.csv"
VERSION_FILE = "model_version.txt"

import pandas as pd
df = pd.read_csv("bias_clean.csv")
print(df["bias"].unique())
print(df.columns.tolist())

#changing bias labels 
df["original_bias"] = df["bias"]

# having 3 classes for training so clean
df["bias"] = df["bias"].replace({
    "leaning-left": "left",
    "leaning-right": "right"
})

# dropping bad rows
df = df.dropna(subset=["page_text", "bias"])

print(df["bias"].value_counts())

#page text
texts = df["page_text"].values
labels = df["bias"].values

#label encoding
le = LabelEncoder()
labels = le.fit_transform(labels)

print(dict(zip(le.classes_, le.transform(le.classes_))))

#train test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
  texts,
  labels,
  df.index,
  test_size=0.2,
  random_state=42,
  stratify=labels
)

#initializing clearml
task = Task.init(
    project_name="Bias Detection",
    task_name="bert_vs_distilbert"
)

#tokenizing text
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_)
)
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }