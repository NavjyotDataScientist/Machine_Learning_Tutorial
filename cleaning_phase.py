"""
cleaning_phase.py

A reusable data cleaning module implementing common cleaning steps as a scikit-learn style pipeline.
Use this module to fit cleaning transforms on training data and apply them to new data consistently.

Example:
    from cleaning_phase import load_data, build_cleaning_pipeline, save_pipeline
    df = load_data("data/raw/data.csv")
    numeric_cols = ["age", "income", "score"]
    categorical_cols = ["gender", "city"]
    pipeline = build_cleaning_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(df)
    df_clean = pipeline.transform(df)
    save_pipeline(pipeline, "artifacts/cleaning_pipeline.joblib")
"""
from typing import List, Optional, Union, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# ---------- Utility functions ----------

def load_data(path: str, **kwargs) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame. Pass additional kwargs to pd.read_csv."""
    return pd.read_csv(path, **kwargs)

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with counts and percent missing per column."""
    miss = df.isna().sum()
    pct = miss / len(df) * 100
    return pd.DataFrame({"missing_count": miss, "missing_pct": pct}).sort_values("missing_pct", ascending=False)

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first") -> pd.DataFrame:
    """Drop duplicate rows. Returns a new DataFrame."""
    return df.drop_duplicates(subset=subset, keep=keep)

def parse_dates(df: pd.DataFrame, date_columns: List[str], fmt: Optional[str] = None, errors: str = "coerce") -> pd.DataFrame:
    """Convert columns to datetime dtype in-place and return df. If fmt provided, used by to_datetime."""
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format=fmt, errors=errors)
    return df

def detect_outliers_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.Series:
    """Return boolean Series indicating rows that have any column with abs(zscore) > threshold."""
    subset = df[columns].select_dtypes(include=[np.number]).copy()
    z = (subset - subset.mean()) / subset.std(ddof=0)
    mask = (z.abs() > threshold).any(axis=1)
    return mask

def winsorize_series(s: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.Series:
    """Return a copied Series capped at the given quantiles."""
    lower = s.quantile(lower_quantile)
    upper = s.quantile(upper_quantile)
    return s.clip(lower, upper)

# ---------- Custom transformers ----------

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Selects columns and returns a DataFrame (not a numpy array). Useful inside pipelines that expect DataFrames."""
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.columns]
        else:
            return pd.DataFrame(X, columns=self.columns)

class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop a list of columns and return DataFrame."""
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns, errors="ignore")

class FillIndicatorTransformer(BaseEstimator, TransformerMixin):
    """Add indicator columns for missingness of selected columns."""
    def __init__(self, columns: List[str], suffix: str = "_was_missing"):
        self.columns = columns
        self.suffix = suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[f"{col}{self.suffix}"] = X[col].isna().astype(int)
        return X

# ---------- Pipeline builder ----------

def build_cleaning_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    date_cols: Optional[List[str]] = None,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "constant",
    categorical_fill_value: str = "MISSING",
    scale_numeric: bool = True,
    one_hot_encode: bool = True,
) -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that:
    - imputes numeric columns (median/mean/most_frequent)
    - imputes categorical columns (fill with 'MISSING' by default)
    - optionally scales numeric columns and one-hot encodes categoricals
    The returned object follows sklearn Transformer API (fit/transform).
    """

    transformers = []

    # Numeric pipeline
    numeric_steps = []
    if numeric_impute_strategy is not None:
        numeric_steps.append(("imputer", SimpleImputer(strategy=numeric_impute_strategy)))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    transformers.append(("num", numeric_pipe, numeric_cols))

    # Categorical pipeline
    cat_steps = []
    if categorical_impute_strategy is not None:
        cat_steps.append(("imputer", SimpleImputer(strategy=categorical_impute_strategy, fill_value=categorical_fill_value)))
    if one_hot_encode:
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)))
    cat_pipe = Pipeline(cat_steps) if cat_steps else "passthrough"
    transformers.append(("cat", cat_pipe, categorical_cols))

    # Date columns: keep as-is (caller can parse dates before using pipeline)
    if date_cols:
        # We keep date columns untouched in the transformer; user may extract features first.
        transformers.append(("date_passthrough", "passthrough", date_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    return ct

# ---------- Persistence ----------

def save_pipeline(pipeline, path: str) -> None:
    """Save pipeline or object with joblib; ensures directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)

def load_pipeline(path: str):
    """Load a joblib-saved pipeline."""
    return joblib.load(path)

# ---------- Example helper to run an end-to-end cleaning on a DataFrame ----------

def fit_transform_dataframe(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    date_cols: Optional[List[str]] = None,
    **pipeline_kwargs
) -> (pd.DataFrame, ColumnTransformer):
    """
    Fit a cleaning pipeline to df and return transformed DataFrame and the fitted pipeline.
    The transformed result is a pandas DataFrame with named columns where possible.
    """
    df_proc = df.copy()
    # Optional: parse any date columns before pipeline (if given)
    if date_cols:
        parse_dates(df_proc, date_cols)

    pipeline = build_cleaning_pipeline(numeric_cols, categorical_cols, date_cols=date_cols, **pipeline_kwargs)
    pipeline.fit(df_proc)

    # Transform
    arr = pipeline.transform(df_proc)

    # Build meaningful column names for numpy output from ColumnTransformer:
    out_cols = []
    for name, trans, cols in pipeline.transformers_:
        if trans == "passthrough":
            out_cols.extend(cols)
            continue
        # If the transformer is a Pipeline, get the last step
        last = trans
        # If OneHotEncoder exists inside, derive feature names
        if isinstance(trans, Pipeline):
            # if OneHotEncoder is the last step or in pipeline, handle it
            if any(isinstance(step[1], OneHotEncoder) for step in trans.steps):
                # find the OneHotEncoder
                for step_name, step_est in trans.steps:
                    if isinstance(step_est, OneHotEncoder):
                        enc = step_est
                        break
                # get feature names
                try:
                    categories = enc.categories_
                    # produce names
                    for i, col in enumerate(cols):
                        cats = categories[i]
                        names = [f"{col}__{str(c)}" for c in cats]
                        out_cols.extend(names)
                except Exception:
                    # fallback
                    out_cols.extend(cols)
            else:
                # numerical scaler etc. keep original names
                out_cols.extend(cols)
        else:
            # unknown transformer; fallback
            out_cols.extend(cols)

    # If result is 1D vector (single column), ensure 2D
    arr = np.atleast_2d(arr)
    df_out = pd.DataFrame(arr, columns=out_cols)
    return df_out, pipeline

# ---------- Basic CLI usage ----------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run cleaning pipeline on a CSV file")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--out", required=False, help="Path to output cleaned CSV")
    parser.add_argument("--pipeline", required=False, help="Path to save pipeline joblib")
    args = parser.parse_args()

    df = load_data(args.input)
    # Example heuristics — in practice pass these explicitly or load from metadata
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Remove obvious ID/time columns from encoding heuristics if desired
    if "id" in numeric_cols:
        numeric_cols.remove("id")

    print("Columns detected:")
    print("Numeric:", numeric_cols)
    print("Categorical:", categorical_cols)

    df_clean, pipeline = fit_transform_dataframe(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df_clean.to_csv(args.out, index=False)
        print(f"Wrote cleaned data to {args.out}")
    if args.pipeline:
        save_pipeline(pipeline, args.pipeline)
        print(f"Saved pipeline to {args.pipeline}")