# ==========================================
# MODEL TRAINING & BASELINE MODELING
# ==========================================

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# -----------------------------
# INPUTS FROM PREVIOUS PHASE
# -----------------------------
# Assumes these variables already exist:
# X_train_processed, X_test_processed
# y_train, y_test

TASK_TYPE = "classification"  # change to "regression" if needed

# -----------------------------
# Classification Baselines
# -----------------------------
if TASK_TYPE == "classification":

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else None

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        })

    results_df = pd.DataFrame(results)
    display(results_df)

# -----------------------------
# Regression Baselines
# -----------------------------
if TASK_TYPE == "regression":

    models = {
        "Linear Regression": LinearRegression()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "R2": r2_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    display(results_df)

print("\nBaseline modeling completed successfully.")
