from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


@dataclass
class AtomicCondition:
    feature: str
    op: str  # "<=" or ">"
    threshold: float


@dataclass
class ICRule:
    id: int
    antecedent_clauses: List[List[AtomicCondition]]  # disjunction of conjunctions
    consequent: List[Dict[str, Any]]
    support: int
    error_fraction: float
    tree_index: int

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "antecedent_clauses": [
                [asdict(cond) for cond in clause] for clause in self.antecedent_clauses
            ],
            "consequent": self.consequent,
            "support": self.support,
            "error_fraction": self.error_fraction,
            "tree_index": self.tree_index,
        }


def _build_rf_for_errors(X: np.ndarray, error_labels: np.ndarray, config: Dict[str, Any]) -> RandomForestClassifier:
    ic_cfg = config.get("ic_generation", {})
    n_estimators = int(ic_cfg.get("rf_n_estimators", 50))
    max_depth = ic_cfg.get("rf_max_depth")
    if max_depth is not None:
        max_depth = int(max_depth)
    min_samples_leaf = int(ic_cfg.get("rf_min_samples_leaf", 5))
    random_state = int(ic_cfg.get("rf_random_state", config.get("seed", 42)))

    # Optional: Fehlerklasse stärker gewichten, um Fehlermuster im RF zu betonen
    error_class_weight = float(ic_cfg.get("error_class_weight", 1.0))
    class_weight = None
    if error_class_weight != 1.0:
        # 0 = korrekt, 1 = Fehler
        class_weight = {0: 1.0, 1: error_class_weight}

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight,
    )
    rf.fit(X, error_labels)
    return rf


def _extract_rules_from_tree(
    tree,
    tree_index: int,
    feature_names: List[str],
    min_error_fraction: float,
    min_support: int,
    start_rule_id: int,
) -> List[ICRule]:
    rules: List[ICRule] = []

    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value  # shape (n_nodes, 1, 2) for binary error/no-error

    path: List[AtomicCondition] = []

    def recurse(node: int):
        nonlocal path, rules

        is_leaf = children_left[node] == children_right[node]
        if is_leaf:
            counts = value[node][0]
            total = counts.sum()
            if total <= 0:
                return
            error_count = counts[1] if counts.shape[0] > 1 else 0.0
            error_fraction = float(error_count / total)
            if total >= min_support and error_fraction >= min_error_fraction:
                rule_id = start_rule_id + len(rules)
                consequent = [
                    {
                        "type": "flag_unreliable",
                        "description": "Prediction in this region is frequently erroneous",
                    }
                ]
                rules.append(
                    ICRule(
                        id=rule_id,
                        antecedent_clauses=[list(path)],
                        consequent=consequent,
                        support=int(total),
                        error_fraction=error_fraction,
                        tree_index=tree_index,
                    )
                )
            return

        feat_idx = feature[node]
        thr = float(threshold[node])
        feat_name = feature_names[feat_idx]

        # Left child: feature <= threshold
        path.append(AtomicCondition(feature=feat_name, op="<=", threshold=thr))
        recurse(children_left[node])
        path.pop()

        # Right child: feature > threshold
        path.append(AtomicCondition(feature=feat_name, op=">", threshold=thr))
        recurse(children_right[node])
        path.pop()

    recurse(0)
    return rules


def generate_ic_rules_for_residual(
    X_residual: np.ndarray,
    y_residual: np.ndarray,
    y_pred_residual: np.ndarray,
    config: Dict[str, Any],
    ic_rules_dir: Path,
) -> List[ICRule]:
    if X_residual is None or y_residual is None or y_pred_residual is None:
        return []

    error_labels = (y_residual != y_pred_residual).astype(int)

    ic_cfg = config.get("ic_generation", {})
    min_error_fraction = float(ic_cfg.get("min_error_fraction", 0.6))
    min_support = int(ic_cfg.get("min_support", 10))
    miner = str(ic_cfg.get("miner", "rf")).lower()

    feature_names = [f"f{j}" for j in range(X_residual.shape[1])]

    all_rules: List[ICRule] = []
    rf_stats: Dict[str, Any] | None = None

    if miner == "rf":
        rf = _build_rf_for_errors(X_residual, error_labels, config)
        start_rule_id = 0
        for tree_index, estimator in enumerate(rf.estimators_):
            tree = estimator.tree_
            rules = _extract_rules_from_tree(
                tree,
                tree_index=tree_index,
                feature_names=feature_names,
                min_error_fraction=min_error_fraction,
                min_support=min_support,
                start_rule_id=start_rule_id,
            )
            all_rules.extend(rules)
            start_rule_id = len(all_rules)

        depths = [int(getattr(est.tree_, "max_depth", 0)) for est in rf.estimators_]
        node_counts = [int(getattr(est.tree_, "node_count", 0)) for est in rf.estimators_]
        rf_stats = {
            "n_estimators": len(depths),
            "max_depth_max": max(depths) if depths else 0,
            "max_depth_mean": float(np.mean(depths)) if depths else 0.0,
            "node_count_mean": float(np.mean(node_counts)) if node_counts else 0.0,
        }

        depths = [int(getattr(est.tree_, "max_depth", 0)) for est in rf.estimators_]
        node_counts = [int(getattr(est.tree_, "node_count", 0)) for est in rf.estimators_]
        rf_stats = {
            "n_estimators": len(depths),
            "max_depth_max": max(depths) if depths else 0,
            "max_depth_mean": float(np.mean(depths)) if depths else 0.0,
            "node_count_mean": float(np.mean(node_counts)) if node_counts else 0.0,
        }

        # Optional: Baumplots einzelner RF-Bäume zur Visualisierung der IC-Struktur
        if bool(ic_cfg.get("export_tree_plots", False)):
            try:
                from sklearn.tree import plot_tree  # type: ignore[import]
                import matplotlib.pyplot as plt  # type: ignore[import]

                max_plots = int(ic_cfg.get("max_tree_plots", 3))
                trees_dir = ic_rules_dir / "trees"
                trees_dir.mkdir(parents=True, exist_ok=True)

                for tree_index, estimator in enumerate(rf.estimators_[:max_plots]):
                    fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore[call-arg]
                    plot_tree(
                        estimator,
                        feature_names=feature_names,
                        filled=True,
                        ax=ax,
                    )
                    fig.tight_layout()
                    plot_path = trees_dir / f"tree_{tree_index}.png"
                    fig.savefig(plot_path, dpi=150)  # type: ignore[arg-type]
                    plt.close(fig)  # type: ignore[call-arg]
            except Exception:
                # Plot-Erzeugung ist optional; bei Fehlern wird sie still übersprungen.
                pass

    elif miner == "cluster":
        # KMeans-Clustering auf Fehlersamples, dann Bounding-Box-Regeln mit zielgerichteter Korrekturklasse
        error_mask = error_labels == 1
        num_error = int(error_mask.sum())
        if num_error > 1:
            X_err = X_residual[error_mask]
            n_clusters = int(ic_cfg.get("cluster_n_clusters", 10))
            max_features_per_rule = int(ic_cfg.get("cluster_max_features", 5))
            min_expected_gain = int(ic_cfg.get("min_expected_gain", 1))
            n_clusters = max(1, min(n_clusters, num_error))

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=int(ic_cfg.get("rf_random_state", config.get("seed", 42))),
            )
            labels = kmeans.fit_predict(X_err)

            start_rule_id = 0
            for cluster_idx in range(n_clusters):
                cluster_mask = labels == cluster_idx
                if not np.any(cluster_mask):
                    continue
                Xc = X_err[cluster_mask]
                mins = Xc.min(axis=0)
                maxs = Xc.max(axis=0)

                # Berechne Support und Fehleranteil innerhalb der Bounding-Box
                in_box = np.all((X_residual >= mins) & (X_residual <= maxs), axis=1)
                total = int(in_box.sum())
                if total == 0:
                    continue
                error_count = int(error_labels[in_box].sum())
                error_fraction = float(error_count / total)
                if total < min_support or error_fraction < min_error_fraction:
                    continue

                # Bestimme typische Zielklasse in dieser Fehler-Region auf Basis der wahren Residual-Labels
                in_box_errors = in_box & error_mask
                if not np.any(in_box_errors):
                    continue
                y_err_box = y_residual[in_box_errors]
                values, counts = np.unique(y_err_box, return_counts=True)
                target_class = int(values[int(np.argmax(counts))])
                tp_gain = int(np.sum(y_err_box == target_class))
                if tp_gain < min_expected_gain:
                    # Diese Regel hätte im Residual keine nennenswerte positive Netto-Wirkung
                    continue

                # Wähle wenige Features mit größter Spannweite für die Regel
                ranges = maxs - mins
                feature_indices = np.argsort(ranges)[::-1]
                chosen_features = [idx for idx in feature_indices if ranges[idx] > 0][:max_features_per_rule]
                clause: List[AtomicCondition] = []
                for feat_idx in chosen_features:
                    feat_name = feature_names[feat_idx]
                    clause.append(AtomicCondition(feature=feat_name, op=">", threshold=float(mins[feat_idx])))
                    clause.append(AtomicCondition(feature=feat_name, op="<=", threshold=float(maxs[feat_idx])))

                if not clause:
                    continue

                rule_id = start_rule_id + len(all_rules)
                consequent = [
                    {
                        "type": "set_class",
                        "target_class": target_class,
                        "description": "Set prediction to majority true class in this clustered error region",
                    }
                ]
                all_rules.append(
                    ICRule(
                        id=rule_id,
                        antecedent_clauses=[clause],
                        consequent=consequent,
                        support=total,
                        error_fraction=error_fraction,
                        tree_index=cluster_idx,
                    )
                )

    else:
        raise ValueError(f"Unknown IC miner '{miner}', expected 'rf' or 'cluster'")

    ic_rules_dir.mkdir(parents=True, exist_ok=True)
    json_path = ic_rules_dir / "ic_rules_residual.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([rule.to_serializable() for rule in all_rules], f, indent=2)

    summary_path = ic_rules_dir / "ic_rules_residual_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for rule in all_rules:
            clause_strs = []
            for clause in rule.antecedent_clauses:
                cond_str = " AND ".join(
                    f"{c.feature} {c.op} {c.threshold:.4f}" for c in clause
                )
                clause_strs.append(f"({cond_str})")
            antecedent_str = " OR ".join(clause_strs)
            line = (
                f"IC {rule.id}: IF {antecedent_str} THEN flag_unreliable "
                f"[support={rule.support}, error_fraction={rule.error_fraction:.3f}]\n"
            )
            f.write(line)

    if rf_stats is not None:
        stats_path = ic_rules_dir / "ic_rf_stats.json"
        with stats_path.open("w", encoding="utf-8") as f_stats:
            json.dump(rf_stats, f_stats, indent=2)

    return all_rules
