import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = 'Customer Churn.csv'
df = pd.read_csv(data_path)

# Display the first few rows to understand the structure
display(df.head())

# 1. Data Overview
print("Dataset Information:\n")
df.info()

print("\nSummary Statistics:\n")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

# 2. Univariate Analysis
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Churn', palette='viridis')
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Numerical feature distribution (e.g., 'MonthlyCharges')
plt.figure(figsize=(8, 5))
sns.histplot(df['MonthlyCharges'], kde=True, color='blue', bins=30)
plt.title('Monthly Charges Distribution')
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')
plt.show()

# 3. Bivariate Analysis
# Churn vs Monthly Charges
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='coolwarm')
plt.title('Churn vs Monthly Charges')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')
plt.show()

# Churn vs Tenure
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='Churn', y='tenure', palette='muted')
plt.title('Churn vs Tenure')
plt.xlabel('Churn')
plt.ylabel('Tenure')
plt.show()

# 4. Correlation Heatmap (for numerical features)
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
