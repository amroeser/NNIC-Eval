from typing import Any, Dict, List, Tuple

import numpy as np

from nnic.ic_generation.rule_extraction import AtomicCondition, ICRule


def _build_feature_index(num_features: int) -> Dict[str, int]:
    return {f"f{j}": j for j in range(num_features)}


def _condition_holds(cond: AtomicCondition, x: np.ndarray, feature_index: Dict[str, int]) -> bool:
    idx = feature_index.get(cond.feature)
    if idx is None:
        return False
    value = x[idx]
    if cond.op == "<=":
        return value <= cond.threshold
    if cond.op == ">":
        return value > cond.threshold
    return False


def _rule_triggers(rule: ICRule, x: np.ndarray, feature_index: Dict[str, int]) -> bool:
    for clause in rule.antecedent_clauses:
        clause_ok = True
        for cond in clause:
            if not _condition_holds(cond, x, feature_index):
                clause_ok = False
                break
        if clause_ok:
            return True
    return False


def apply_ic_rules(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rules: List[ICRule],
    config: Dict[str, Any],
    y_train: np.ndarray | None = None,
) -> Dict[str, Any]:
    if len(rules) == 0:
        return {
            "mode": "none",
            "y_pred_ic": y_pred,
            "flagged": np.zeros_like(y_pred, dtype=bool),
        }

    app_cfg = config.get("ic_application", {})
    mode = str(app_cfg.get("mode", "flagging")).lower()
    # Nur auf Fehlklassifikationen anwenden? Standard: True, damit Acc_post >= Acc_pre bleibt
    errors_only = bool(app_cfg.get("errors_only", True))

    n_samples, n_features = X.shape
    feature_index = _build_feature_index(n_features)

    flagged = np.zeros(n_samples, dtype=bool)
    y_pred_ic = np.array(y_pred, copy=True)
    error_mask = y_pred != y_true

    if mode == "correction":
        if y_train is not None and y_train.size > 0:
            values, counts = np.unique(y_train, return_counts=True)
        else:
            values, counts = np.unique(y_pred, return_counts=True)
        fallback_class = values[int(np.argmax(counts))]
    else:
        fallback_class = None

    collision_policy = str(app_cfg.get("collision_policy", "first_match")).lower()

    for i in range(n_samples):
        # Korrekt klassifizierte Samples werden im Korrekturmodus nicht verändert,
        # damit die Gesamtgenauigkeit nicht sinkt.
        if mode == "correction" and errors_only and not error_mask[i]:
            continue

        x = X[i]

        # Sammle alle Regeln, die für dieses Sample feuern
        triggered_indices: List[int] = []
        for idx, rule in enumerate(rules):
            if _rule_triggers(rule, x, feature_index):
                triggered_indices.append(idx)

        if not triggered_indices:
            continue

        # Wähle gemäß Kollisions-Policy genau eine Regel aus
        if collision_policy == "max_support":
            chosen_idx = max(
                triggered_indices,
                key=lambda j: getattr(rules[j], "support", 0),
            )
        elif collision_policy == "max_error_fraction":
            chosen_idx = max(
                triggered_indices,
                key=lambda j: getattr(rules[j], "error_fraction", 0.0),
            )
        else:  # "first_match" (Standard)
            chosen_idx = triggered_indices[0]

        chosen_rule = rules[chosen_idx]
        flagged[i] = True

        if mode == "correction":
            # Bevorzugt eine regel-spezifische Zielklasse, falls vorhanden
            target_class = None
            for cons in getattr(chosen_rule, "consequent", []) or []:
                if cons.get("type") == "set_class" and "target_class" in cons:
                    target_class = cons["target_class"]
                    break

            if target_class is not None:
                y_pred_ic[i] = target_class
            elif fallback_class is not None:
                y_pred_ic[i] = fallback_class

    return {
        "mode": mode,
        "y_pred_ic": y_pred_ic,
        "flagged": flagged,
    }
