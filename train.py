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

#