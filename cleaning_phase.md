# Data Cleaning phase — code + explanations

This file contains a reusable, production-friendly Data Cleaning phase you can copy into a repository. It includes:

- A short conceptual overview of common cleaning tasks.
- Ready-to-use Python code (see cleaning_phase.py) that implements a cleaning pipeline using pandas and scikit-learn style transformers.
- Examples and usage snippets.
- A short checklist and recommendations for publishing to GitHub.

Prerequisites
- Python 3.8+
- pandas, numpy, scikit-learn, joblib (see requirements.txt)
- Basic familiarity with pandas DataFrames

Overview — what belongs in the Cleaning phase
- Data ingestion / schema checks and lightweight validation
- Handling missing values (per-column strategy)
- Correcting types (numeric, categorical, datetime)
- Removing or flagging duplicates
- Handling outliers (detection, winsorizing, capping, or removal)
- Standardizing strings (trimming, lowercasing)
- Encoding categorical variables (target or model-appropriate encoding)
- Scaling numeric variables (if models require it)
- Creating simple derived features needed for modeling (e.g., age from dob)
- Persisting the cleaning rules (pipeline) so training and production use identical steps

Design principles
- Prefer explicit column lists and per-column strategies for reproducibility.
- Keep the pipeline deterministic and save it (joblib) for reuse.
- Separate "fit" (learn imputation statistics, categories, scalers) from "transform" (apply to new data).
- Always save metadata/versions (columns expected, dtypes, imputers, etc.).

Files in this example
- cleaning_phase.py — Python module with functions and a scikit-learn style pipeline
- requirements.txt — minimal dependencies

Why use a pipeline?
- Ensures identical transforms between training and inference
- Makes it easy to persist transformations and to integrate inside model training code or production microservices

Example: minimal usage
```python
from cleaning_phase import load_data, build_cleaning_pipeline, save_pipeline

df = load_data("data/raw/data.csv")

numeric_cols = ["age", "income", "score"]
categorical_cols = ["gender", "city"]

pipeline = build_cleaning_pipeline(numeric_cols, categorical_cols)
# Fit the pipeline using training data
pipeline.fit(df)

# Transform training or new data
df_clean = pipeline.transform(df)

# Save pipeline for reuse
save_pipeline(pipeline, "artifacts/cleaning_pipeline.joblib")
```

Notes on missing data strategies
- Numeric columns: median is robust to outliers; mean is fine for symmetric distributions.
- Categorical: imputing to "MISSING" preserves missingness explicitly (useful as a signal).
- Consider adding binary "was_missing" indicator columns if missingness may be informative.

Outliers
- Detect outliers with z-score or IQR. Decide whether to remove, cap (winsorize), or leave based on domain knowledge.
- Avoid dropping rows en masse in datasets with many features; small datasets are especially sensitive.

Categorical encoding
- For tree-based models: use simple ordinal mapping or leave as strings if you use libraries that handle categoricals (or use category dtype).
- For linear models: one-hot encode or use target / frequency encoding with caution (avoid leakage).

Datetime handling
- Always parse dates using sensible timezone handling if required.
- Derive features like year, month, dayofweek, hour, or elapsed time differences as needed.

Reproducibility and metadata
- Save the pipeline (joblib) and a JSON/YAML metadata file listing:
  - expected columns and dtypes
  - version of cleaning script
  - training sample size and date
- Publish a short README explaining how to apply the pipeline to new data.
