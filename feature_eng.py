📘 FEATURE ENGINEERING & PREPROCESSING

(Markdown / README.md content)

🎯 Objective

This phase converts cleaned and explored data into model-ready features, using insights gained from EDA, while strictly avoiding data leakage.

🧠 Why this phase exists

EDA tells us:

Which features are useful

Which are skewed

Which are categorical

Which may cause leakage

Which need scaling or transformation

Feature Engineering applies those decisions in a safe, reproducible way.

🔁 Industry-Standard Order (Very Important)

Select features

Separate target variable

Split data into Train / Test

Fit preprocessing ONLY on training data

Apply same transformations to test data

Output model-ready datasets

🚨 Never fit scalers or encoders on the full dataset

🧩 What is done in this phase
1️⃣ Feature Selection

Drop IDs, leakage columns, non-informative fields

2️⃣ Train–Test Split

Happens before scaling & encoding

Prevents information leakage

3️⃣ Encoding

Categorical → OneHot / Ordinal

4️⃣ Scaling

Numerical → Standard / MinMax / Robust scaling

5️⃣ Pipeline Creation

Ensures reproducibility

Industry & production friendly

🏗 Output of this Phase

X_train_processed

X_test_processed

y_train

y_test

➡️ Ready for model training



# ==========================================
# FEATURE ENGINEERING & PREPROCESSING
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv("cleaned_data.csv")

TARGET = "target_column_name"   # CHANGE THIS

# -----------------------------
# Feature / Target separation
# -----------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# -----------------------------
# Feature type separation
# -----------------------------
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

# -----------------------------
# Train-Test Split (BEFORE scaling!)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() < 10 else None
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# Preprocessing Pipelines
# -----------------------------
numeric_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# -----------------------------
# Fit ONLY on training data
# -----------------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# -----------------------------
# Final outputs
# -----------------------------
print("\nProcessed Training Shape:", X_train_processed.shape)
print("Processed Test Shape:", X_test_processed.shape)

print("\nFeature Engineering & Preprocessing completed successfully.")
