
import pandas as pd
from pathlib import Path

# Load dataset
df = pd.read_csv("data/heart.csv")

print("=" * 50)
print("HEART DISEASE DATASET — INITIAL ANALYSIS")
print("=" * 50)

# Dataset information
print("\n1. Dataset shape:", df.shape)

print("\n2. Column data types:")
print(df.dtypes)

print("\n3. Missing values before cleaning:")
print(df.isnull().sum())

# Convert the target to binary:
# 0 = no recorded disease
# 1–4 = recorded disease
df["target"] = (df["target"] > 0).astype(int)

# Remove rows where the target or required features are missing
df = df.dropna(subset=["target"])

# Fill missing feature values with column medians
df = df.fillna(df.median(numeric_only=True))

print("\n4. Missing values after cleaning:")
print(df.isnull().sum())

print("\n5. Target distribution:")
print(df["target"].value_counts().sort_index())

print("\n6. Target percentages:")
print(
    (df["target"].value_counts(normalize=True) * 100)
    .sort_index()
    .round(2)
)

# Save cleaned data
Path("data").mkdir(exist_ok=True)
df.to_csv("data/heart_cleaned.csv", index=False)

print("\nCleaned dataset saved to data/heart_cleaned.csv")
print("Final shape:", df.shape)
