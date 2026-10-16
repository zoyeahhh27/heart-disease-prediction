
import pandas as pd
from pathlib import Path

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    url,
    names=columns,
    na_values="?"
)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/heart.csv", index=False)

print("Dataset downloaded successfully!")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
