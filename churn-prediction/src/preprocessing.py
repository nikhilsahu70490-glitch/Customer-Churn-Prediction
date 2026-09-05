"""
Preprocessing pipeline for the churn dataset:
  1. Handle missing values
  2. Encode categorical features
  3. Scale numeric features
  4. Split into train/test sets

Kept as reusable functions so the same preprocessing can be applied
consistently in both training and any future inference script.
"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = "Churn"
ID_COL = "customerID"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges sometimes arrives as a string with blank entries for
    brand-new customers (0 tenure). Coerce to numeric, then fill missing
    with 0 since a 0-tenure customer has paid nothing yet.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Label-encodes all categorical (object-type) columns except the ID.
    Returns the encoded dataframe plus a dict of fitted encoders,
    so the same mapping can be reused at inference time.
    """
    df = df.copy()
    encoders = {}
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != ID_COL]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales numeric features to mean 0 / std 1. This matters most for
    Logistic Regression and SVM-style models, which are sensitive to
    feature magnitude; tree-based models (RF, XGBoost) don't strictly
    need it but it doesn't hurt them either.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(path: str, test_size: float = 0.2, random_state: int = 42):
    """Full preprocessing pipeline: load -> clean -> encode -> split -> scale."""
    df = load_data(path)
    df = handle_missing_values(df)
    df, encoders = encode_features(df)

    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]  # already label-encoded: 0 = No, 1 = Yes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoders, scaler
