from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from nnic.evaluation.metrics import basic_classification_metrics


def train_eval_rf_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], RandomForestClassifier]:
    """Trainiert einen Random-Forest-Klassifikator als Baseline auf dem Trainingssplit
    und evaluiert ihn auf dem Testsplit. Hyperparameter werden aus `config["baseline"]["rf"]`
    gelesen, falls vorhanden, ansonsten werden sinnvolle Defaults genutzt.
    """
    baseline_cfg = config.get("baseline", {}).get("rf", {})

    n_estimators = int(baseline_cfg.get("n_estimators", 100))
    max_depth = baseline_cfg.get("max_depth")
    if max_depth is not None:
        max_depth = int(max_depth)
    min_samples_leaf = int(baseline_cfg.get("min_samples_leaf", 1))
    random_state = int(baseline_cfg.get("random_state", config.get("seed", 42)))

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred_rf = clf.predict(X_test)
    metrics_rf = basic_classification_metrics(y_test, y_pred_rf)
    return metrics_rf, clf
