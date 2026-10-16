
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/heart_cleaned.csv")

sns.set_theme(style="whitegrid")

# 1. Target distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="target")
plt.title("Heart Disease Target Distribution")
plt.xlabel("Target (0 = No recorded disease, 1 = Recorded disease)")
plt.ylabel("Number of patients")
plt.tight_layout()
plt.savefig("data/target_distribution.png")
plt.show()

# 2. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="target", bins=20, kde=True)
plt.title("Age Distribution by Target Class")
plt.tight_layout()
plt.savefig("data/age_distribution.png")
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("data/correlation_heatmap.png")
plt.show()

print("EDA completed. Visualizations saved in the data folder.")
