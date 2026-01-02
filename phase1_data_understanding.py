# Phase 1: Data Understanding
# A comprehensive Exploratory Data Analysis (EDA) script for beginners.
# This script loads a dataset and performs initial exploration: structure, statistics,
# missing values, distributions, correlations, and visualizations.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set(style="whitegrid")

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATA_PATH = "../data/raw/my_dataset.csv"  # Update this path to your dataset
TARGET_COLUMN = "target"                  # Change to your actual target column name (e.g., 'income', 'species')

# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please check the path.")

df = pd.read_csv(DATA_PATH)

print("=== Dataset Loaded Successfully ===\n")
print("=== 1. First 5 Rows ===")
print(df.head())

print("\n=== 2. Last 5 Rows ===")
print(df.tail())

# ---------------------------------------------------------
# 2. Basic Information
# ---------------------------------------------------------
print("\n=== 3. Shape of the Dataset ===")
print(df.shape)

print("\n=== 4. Columns and Data Types ===")
print(df.dtypes)

print("\n=== 5. General Info ===")
df.info()

# ---------------------------------------------------------
# 3. Summary Statistics
# ---------------------------------------------------------
print("\n=== 6. Numerical Features - Descriptive Statistics ===")
print(df.describe())

print("\n=== 7. Categorical Features - Descriptive Statistics ===")
print(df.describe(include=["object", "category"]))

# ---------------------------------------------------------
# 4. Missing Values
# ---------------------------------------------------------
print("\n=== 8. Missing Values Count ===")
print(df.isnull().sum())

print("\n=== 9. Missing Values Percentage ===")
missing_percentage = (df.isnull().mean() * 100).round(2)
print(missing_percentage)

# ---------------------------------------------------------
# 5. Target Variable Analysis
# ---------------------------------------------------------
if TARGET_COLUMN in df.columns:
    print(f"\n=== 10. Distribution of Target Variable '{TARGET_COLUMN}' ===")
    print(df[TARGET_COLUMN].value_counts(normalize=True).round(3))

    # Plot target distribution
    try:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df[TARGET_COLUMN], order=df[TARGET_COLUMN].value_counts().index)
        plt.title(f"Target Variable Distribution - {TARGET_COLUMN}")
        plt.xlabel(TARGET_COLUMN)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plot skipped (non-interactive environment): {e}")
else:
    print(f"\nWarning: Target column '{TARGET_COLUMN}' not found in dataset.")

# ---------------------------------------------------------
# 6. Unique Values (Cardinality)
# ---------------------------------------------------------
print("\n=== 11. Number of Unique Values per Column ===")
print(df.nunique())

# ---------------------------------------------------------
# 7. Correlation Analysis (Numerical Features)
# ---------------------------------------------------------
numerical_cols = df.select_dtypes(include=np.number).columns
if len(numerical_cols) > 1:
    print("\n=== 12. Top 10 Correlations (Absolute) ===")
    corr = df[numerical_cols].corr()
    corr_unstack = corr.unstack().sort_values(ascending=False, key=abs).drop_duplicates()
    print(corr_unstack.head(10))

    # Heatmap
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap (Numerical Features)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plot skipped: {e}")

# ---------------------------------------------------------
# 8. Distribution of Numerical Features
# ---------------------------------------------------------
print(f"\n=== 13. Distributions for {len(numerical_cols)} Numerical Columns ===")
for col in numerical_cols:
    try:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plot skipped for {col}: {e}")

# ---------------------------------------------------------
# 9. Outlier Detection via Boxplots
# ---------------------------------------------------------
print("\n=== 14. Boxplots for Numerical Columns (Outliers) ===")
for col in numerical_cols:
    try:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col} (Outlier Detection)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plot skipped for {col}: {e}")

# ---------------------------------------------------------
# 10. Categorical Features Analysis
# ---------------------------------------------------------
categorical_cols = df.select_dtypes(include="object").columns
print(f"\n=== 15. Value Counts for {len(categorical_cols)} Categorical Columns ===")
for col in categorical_cols:
    print(f"\n--- {col} (Top 10 values) ---")
    print(df[col].value_counts().head(10))

    # Bar plot of top 10
    try:
        top_n = df[col].value_counts().head(10)
        plt.figure(figsize=(10, 5))
        top_n.plot(kind="bar")
        plt.title(f"Top 10 Values in {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plot skipped for {col}: {e}")

print("\n=== Data Understanding Phase Complete! ===")
print("Next recommended steps:")
print("- Data cleaning (handle missing values, outliers)")
print("- Feature engineering and encoding")
print("- Modeling preparation")