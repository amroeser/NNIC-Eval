from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _determine_positive_class(y: np.ndarray, config: Dict[str, Any]) -> Any:
    eval_cfg = config.get("evaluation", {})
    if "positive_class" in eval_cfg:
        return eval_cfg["positive_class"]
    unique = np.unique(y)
    return unique.max()


def compute_classification_flags(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = y_true == y_pred
    error = ~correct

    flags: Dict[str, np.ndarray] = {
        "correct": correct,
        "error": error,
    }

    unique_labels = np.unique(y_true)
    if unique_labels.size == 2:
        pos = _determine_positive_class(y_true, config)
        neg = unique_labels[0] if unique_labels[1] == pos else unique_labels[1]
        tp = (y_true == pos) & (y_pred == pos)
        tn = (y_true == neg) & (y_pred == neg)
        fp = (y_true == neg) & (y_pred == pos)
        fn = (y_true == pos) & (y_pred == neg)
        flags.update({
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })

    return flags


def extract_and_export_misclassifications(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    config: Dict[str, Any],
    results_dir: Path,
) -> Dict[str, Any]:
    n_samples, n_features = X.shape
    indices = np.arange(n_samples)
    flags = compute_classification_flags(y_true, y_pred, config)

    error_mask = flags["error"]
    error_indices = indices[error_mask]

    df_dict: Dict[str, Any] = {
        "index": indices,
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": flags["correct"],
    }

    for name in ("tp", "tn", "fp", "fn"):
        mask = flags.get(name)
        if mask is not None:
            df_dict[name] = mask

    for j in range(n_features):
        df_dict[f"f{j}"] = X[:, j]

    df = pd.DataFrame(df_dict)

    csv_path = results_dir / f"misclassifications_{split_name}.csv"
    df.to_csv(csv_path, index=False)

    return {
        "csv_path": csv_path,
        "error_indices": error_indices,
        "flags": flags,
    }
