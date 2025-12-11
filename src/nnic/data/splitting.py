from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split


def train_val_residual_test_split(X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    splits_cfg = config.get("splits", {})
    test_size = float(splits_cfg.get("test_size", 0.05))
    val_size = float(splits_cfg.get("val_size", 0.0))
    residual_size = float(splits_cfg.get("residual_size", 0.0))
    random_state = int(splits_cfg.get("random_state", 42))
    stratify_flag = bool(splits_cfg.get("stratify", True))

    stratify = y if stratify_flag else None

    X_train_val_res, X_test, y_train_val_res, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    remaining = 1.0 - test_size
    if remaining <= 0:
        raise ValueError("test_size must be less than 1.0")

    val_prop = val_size / remaining if val_size > 0 else 0.0
    residual_prop = residual_size / remaining if residual_size > 0 else 0.0

    X_train = X_train_val_res
    y_train = y_train_val_res
    X_val = None
    y_val = None
    X_residual = None
    y_residual = None

    if val_prop + residual_prop > 0:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_train_val_res,
            y_train_val_res,
            test_size=(val_prop + residual_prop),
            random_state=random_state,
            stratify=y_train_val_res if stratify is not None else None,
        )

        if residual_prop > 0:
            # Sonderfall: kein Validierungsset, nur Residual-Set
            if val_prop == 0.0:
                X_residual, y_residual = X_temp, y_temp
                X_val, y_val = None, None
            else:
                if val_prop + residual_prop <= 0:
                    raise ValueError("Invalid val_size and residual_size configuration")
                rel_residual = residual_prop / (val_prop + residual_prop)
                X_val, X_residual, y_val, y_residual = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=rel_residual,
                    random_state=random_state,
                    stratify=y_temp if stratify is not None else None,
                )
        else:
            X_val = X_temp
            y_val = y_temp

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "residual": (X_residual, y_residual),
        "test": (X_test, y_test),
    }
