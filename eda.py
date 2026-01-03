# Note: this is just for learning outcome as many outcome can comes as per experience.

# ================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.3f}'.format)
sns.set_style("whitegrid")

# ----------------
# Load cleaned data
# ----------------
df = pd.read_csv("cleaned_data.csv")

TARGET = "target_column_name"   # CHANGE THIS

print("="*80)
print("DATASET SHAPE:", df.shape)
print("="*80)
display(df.head())

# ----------------
# Feature separation
# ----------------
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

if TARGET in numerical_features:
    numerical_features.remove(TARGET)

print("\nNumerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

# ----------------
# Target analysis
# ----------------
print("\nTARGET DISTRIBUTION:")
display(df[TARGET].value_counts())
display(df[TARGET].value_counts(normalize=True))

plt.figure(figsize=(6,4))
sns.countplot(x=TARGET, data=df)
plt.title("Target Variable Distribution")
plt.show()

# ----------------
# Univariate analysis - numerical
# ----------------
print("\nNUMERICAL FEATURE SUMMARY:")
display(df[numerical_features].describe())

for col in numerical_features:
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    sns.histplot(df[col], kde=True, ax=ax[0])
    sns.boxplot(x=df[col], ax=ax[1])
    fig.suptitle(f"Univariate Analysis: {col}")
    plt.show()

# ----------------
# Univariate analysis - categorical
# ----------------
for col in categorical_features:
    print(f"\n{col} - Value Counts")
    display(df[col].value_counts().head(10))

    plt.figure(figsize=(6,4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.show()

# ----------------
# Bivariate analysis - numerical vs target
# ----------------
for col in numerical_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[TARGET], y=df[col])
    plt.title(f"{col} vs {TARGET}")
    plt.show()

# ----------------
# Bivariate analysis - categorical vs target
# ----------------
for col in categorical_features:
    cross_tab = pd.crosstab(df[col], df[TARGET], normalize='index')
    cross_tab.plot(kind='bar', stacked=True, figsize=(7,4))
    plt.title(f"{col} vs {TARGET}")
    plt.show()

# ----------------
# Correlation analysis
# ----------------
corr = df[numerical_features + [TARGET]].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix")
plt.show()

# ----------------
# Outlier percentage (IQR method)
# ----------------
print("\nOUTLIER ANALYSIS:")
for col in numerical_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outlier_pct = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).mean()
    print(f"{col}: {outlier_pct:.2%} outliers")

# ----------------
# Skewness check
# ----------------
print("\nSKEWNESS:")
display(df[numerical_features].skew().sort_values(ascending=False))

# ----------------
# Final EDA summary (manual insights)
# ----------------
print("="*80)
print("EDA INSIGHTS (WRITE YOUR OBSERVATIONS HERE)")
print("""
1. Target distribution:
2. Strong predictors observed:
3. High skew features:
4. Correlated features:
5. Categorical patterns:
6. Feature engineering ideas:
""")
print("="*80)
