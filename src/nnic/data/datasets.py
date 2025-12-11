from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, load_digits, load_wine, load_iris, load_breast_cancer, fetch_openml


def _normalize_labels(y: np.ndarray) -> np.ndarray:
    """Map arbitrary class labels to consecutive integers starting at 0."""
    unique = np.unique(y)
    mapping = {label: idx for idx, label in enumerate(unique)}
    return np.vectorize(mapping.get)(y).astype(int)


def load_dataset(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    dataset_cfg = config.get("dataset", {})
    name = dataset_cfg.get("name")

    if name == "digits":
        data = load_digits()
        X, y = data.data, data.target
        return X, _normalize_labels(y)

    if name == "covertype":
        data = fetch_covtype()
        X, y = data.data, data.target
        return X, _normalize_labels(y)

    if name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        return X, _normalize_labels(y)

    if name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        return X, _normalize_labels(y)

    if name == "wine":
        data = load_wine()
        X, y = data.data, data.target
        return X, _normalize_labels(y)

    if name == "openml":
        ds_id = dataset_cfg.get("openml_id")
        if ds_id is None:
            raise ValueError("For dataset 'openml', 'openml_id' must be provided in the config")
        # Load as DataFrame to properly handle categorical features, then one-hot encode
        data = fetch_openml(data_id=int(ds_id), as_frame=True)
        X_df, y = data.data, data.target
        # One-hot encode all categorical columns and ensure numeric dtype
        X_df = pd.get_dummies(X_df, drop_first=False)
        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X = X_df.to_numpy(dtype=float)
        return X, _normalize_labels(np.asarray(y))

    if name == "car":
        csv_path = dataset_cfg.get("csv_path")
        target_column = dataset_cfg.get("target_column")
        if not csv_path or not target_column:
            raise ValueError("For dataset 'car', 'csv_path' and 'target_column' must be provided in the config")
        df = pd.read_csv(csv_path)
        X = df.drop(columns=[target_column]).to_numpy()
        y = df[target_column].to_numpy()
        return X, _normalize_labels(y)

    raise ValueError(f"Unsupported dataset name: {name}")
