import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('your_dataset.csv')

# Display the first few rows
print("First few rows of the data:")
print(data.head())

# Check data types and missing values
print("\nData types and missing values:")
print(data.info())

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical columns:")
print(categorical_cols)
print("\nNumerical columns:")
print(numerical_cols)

# Visualize missing values
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Correlation analysis for numerical features
correlation_matrix = data[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize distributions of numerical features
data[numerical_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features')
plt.show()

# Analyze categorical variables
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=data)
    plt.title(f'Distribution of {col}')
    plt.show()

# Outlier detection for numerical features using boxplots
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
