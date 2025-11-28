import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print("🔍 LOADING HEART DISEASE DATASET...")
df = pd.read_csv('heart_disease_uci.csv')

print(f"Dataset shape: {df.shape}")
print("\nTarget distribution (num - disease severity 0-4):")
print(df['num'].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset preview:")
print(df.head())

# Disease distribution
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
sns.countplot(x='num', data=df)
plt.title('Disease Severity Distribution')
plt.xlabel('Severity (0=No Disease, 4=Critical)')

# Age vs Disease
plt.subplot(2, 2, 2)
sns.boxplot(x='num', y='age', data=df)
plt.title('Age by Disease Severity')

# Cholesterol vs Disease
plt.subplot(2, 2, 3)
sns.boxplot(x='num', y='chol', data=df)
plt.title('Cholesterol by Disease Severity')

# Sex distribution
plt.subplot(2, 2, 4)
sns.countplot(x='sex', hue='num', data=df)
plt.title('Sex vs Disease Severity')
plt.xlabel('Sex (0=Female, 1=Male)')

plt.tight_layout()
plt.show()

print("\n✅ EDA COMPLETE!")
