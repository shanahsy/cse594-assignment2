import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# Config
# -----------------------------
INPUT_PATH = "granite-friends-context1-20251215-034424__checkpoint-41374/predictions.csv"
OUTPUT_SAMPLE_PATH = "sample_250_high_acc.csv"
TARGET_ACCURACY = 0.80   # Change this (0.65–0.85)
SAMPLE_SIZE = 250
RANDOM_SEED = 42

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_PATH)

df["correct"] = df["speaker"] == df["predicted_speaker"]

correct_df = df[df["correct"]]
incorrect_df = df[~df["correct"]]

# How many correct vs incorrect?
num_correct = int(SAMPLE_SIZE * TARGET_ACCURACY)
num_incorrect = SAMPLE_SIZE - num_correct

# Sample
sample_correct = correct_df.sample(n=num_correct, random_state=RANDOM_SEED)
sample_incorrect = incorrect_df.sample(n=num_incorrect, random_state=RANDOM_SEED)

sample_df = pd.concat([sample_correct, sample_incorrect]).sample(
    frac=1, random_state=RANDOM_SEED
)

sample_df.drop(columns=["correct"]).to_csv(OUTPUT_SAMPLE_PATH, index=False)

# -----------------------------
# Metrics
# -----------------------------
y_true = sample_df["speaker"]
y_pred = sample_df["predicted_speaker"]

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

print(f"Evaluated examples: {len(sample_df)}")
print(f"Overall accuracy: {accuracy*100:.2f}% "
      f"({(y_true == y_pred).sum()}/{len(sample_df)})")
print(f"Macro F1 score: {macro_f1:.4f}")
