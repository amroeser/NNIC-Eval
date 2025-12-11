from pathlib import Path
import json
import time

import torch.nn as nn

from nnic.config.loader import load_config
from nnic.data.datasets import load_dataset
from nnic.data.splitting import train_val_residual_test_split
from nnic.evaluation.metrics import basic_classification_metrics, model_predict
from nnic.experiment.utils import prepare_output_dirs, set_global_seeds
from nnic.ic_application.apply_ic import apply_ic_rules
from nnic.ic_generation.rule_extraction import generate_ic_rules_for_residual
from nnic.ic_generation.io import load_ic_rules_from_json
from nnic.misclassification.extraction import extract_and_export_misclassifications
from nnic.training.nn_trainer import train_nn_model
from nnic.baselines.rf_baseline import train_eval_rf_baseline


def _execute_experiment(config: dict) -> dict:
    """Core experiment execution: takes a config dict and returns the summary.

    Side effects:
      - erstellt Output-Verzeichnisse
      - schreibt summary.json im jeweiligen results-Ordner
      - schreibt IC-Regeln und Misclassification-CSVs
    """
    seed = config.get("seed", 42)
    set_global_seeds(seed)

    base_dir = config.get("base_output_dir", ".")
    experiment_name = config.get("experiment_name", "experiment")
    output_dirs = prepare_output_dirs(base_dir, experiment_name)

    X, y = load_dataset(config)
    splits = train_val_residual_test_split(X, y, config)

    print(f"Loaded config for experiment '{experiment_name}'")
    print(f"Seed: {seed}")
    print(f"Output directories: {output_dirs}")
    for split_name, (X_split, y_split) in splits.items():
        if X_split is not None and y_split is not None:
            print(f"{split_name}: {X_split.shape}, {y_split.shape}")
        else:
            print(f"{split_name}: None")

    t0_train = time.perf_counter()
    model, device, history = train_nn_model(splits, config, output_dirs)
    t1_train = time.perf_counter()

    X_val, y_val = splits.get("val", (None, None))
    batch_size = int(config.get("training", {}).get("batch_size", 64))

    metrics_val = None
    if X_val is not None and y_val is not None:
        y_pred_val = model_predict(model, X_val, batch_size=batch_size, device=device)
        metrics_val = basic_classification_metrics(y_val, y_pred_val)

    X_test, y_test = splits["test"]
    y_pred_test = model_predict(model, X_test, batch_size=batch_size, device=device)
    metrics_pre = basic_classification_metrics(y_test, y_pred_test)

    print("Test metrics (NN without ICs):")
    for key, value in metrics_pre.items():
        print(f"  {key}: {value}")

    results_dir = output_dirs["results"]
    ic_rules_dir = output_dirs["ic_rules"]

    # Optional: Random-Forest-Baseline-Klassifikator auf Train/Test
    baseline_rf_metrics = None
    baseline_cfg = config.get("baseline", {}).get("rf", {})
    if baseline_cfg.get("enabled", False):
        print("Training Random-Forest baseline classifier ...")
        X_train, y_train = splits["train"]
        baseline_rf_metrics, _ = train_eval_rf_baseline(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
        )
        print("Random-Forest baseline test metrics:")
        for key, value in baseline_rf_metrics.items():
            print(f"  {key}: {value}")

    ic_rules = []
    X_residual, y_residual = splits.get("residual", (None, None))
    ic_cfg = config.get("ic_generation", {})
    rules_override = ic_cfg.get("rules_path_override")

    if rules_override:
        override_path = Path(rules_override)
        if not override_path.is_absolute():
            # Relativ zur Projektwurzel interpretieren
            project_root = Path(__file__).resolve().parent.parent
            override_path = project_root / override_path
        if override_path.exists():
            t0_ic_gen = time.perf_counter()
            ic_rules = load_ic_rules_from_json(override_path)
            t1_ic_gen = time.perf_counter()
        else:
            t0_ic_gen = t1_ic_gen = None
    elif X_residual is not None and y_residual is not None:
        t0_ic_gen = time.perf_counter()
        y_pred_residual = model_predict(model, X_residual, batch_size=batch_size, device=device)
        extract_and_export_misclassifications(
            X_residual,
            y_residual,
            y_pred_residual,
            "residual",
            config,
            results_dir,
        )
        ic_rules = generate_ic_rules_for_residual(
            X_residual,
            y_residual,
            y_pred_residual,
            config,
            ic_rules_dir,
        )
        t1_ic_gen = time.perf_counter()
    else:
        t0_ic_gen = t1_ic_gen = None

    extract_and_export_misclassifications(
        X_test,
        y_test,
        y_pred_test,
        "test",
        config,
        results_dir,
    )

    metrics_post = None
    mode = None
    frac_flagged = 0.0

    t0_ic_apply = t1_ic_apply = None

    if ic_rules:
        y_train = splits["train"][1]
        t0_ic_apply = time.perf_counter()
        ic_result = apply_ic_rules(
            X_test,
            y_test,
            y_pred_test,
            ic_rules,
            config,
            y_train=y_train,
        )
        y_pred_ic = ic_result["y_pred_ic"]
        flagged = ic_result["flagged"]
        mode = ic_result["mode"]
        metrics_post = basic_classification_metrics(y_test, y_pred_ic)
        t1_ic_apply = time.perf_counter()

        print(f"Test metrics with ICs (mode={mode}):")
        for key, value in metrics_post.items():
            print(f"  {key}: {value}")

        frac_flagged = float(flagged.mean()) if flagged.size > 0 else 0.0
        print(f"Fraction of test samples flagged by ICs: {frac_flagged:.3f}")
    else:
        print("No IC rules were generated; IC application on test set is skipped.")

    dataset_cfg = config.get("dataset", {})
    raw_dataset_name = dataset_cfg.get("name")

    # For OpenML-based datasets, derive a more specific dataset name so reporting
    # groups adult, bank, and credit-g as separate datasets instead of one 'openml' block.
    if raw_dataset_name == "openml":
        ds_id = dataset_cfg.get("openml_id")
        try:
            ds_id_int = int(ds_id) if ds_id is not None else None
        except (TypeError, ValueError):
            ds_id_int = None

        if ds_id_int == 31:
            dataset_name = "credit-g"
        elif ds_id_int == 1461:
            dataset_name = "bank-marketing"
        elif ds_id_int == 45068:
            dataset_name = "adult"
        elif ds_id_int == 44:
            dataset_name = "spambase"
        elif ds_id_int == 1489:
            dataset_name = "phoneme"
        elif ds_id_int == 1486:
            dataset_name = "nomao"
        elif ds_id_int == 1120:
            dataset_name = "magic-telescope"
        elif ds_id_int == 151:
            dataset_name = "electricity"
        elif ds_id_int is not None:
            dataset_name = f"openml_{ds_id_int}"
        else:
            dataset_name = "openml"
    else:
        dataset_name = raw_dataset_name
    splits_cfg = config.get("splits", {})
    split_counts = {
        "train": int(splits["train"][0].shape[0]) if splits["train"][0] is not None else 0,
        "val": int(splits["val"][0].shape[0]) if splits["val"][0] is not None else 0,
        "residual": int(splits["residual"][0].shape[0]) if splits["residual"][0] is not None else 0,
        "test": int(splits["test"][0].shape[0]) if splits["test"][0] is not None else 0,
    }
    num_params = int(sum(p.numel() for p in getattr(model, "parameters", lambda: [])()))

    timing = {
        "train_seconds": float(t1_train - t0_train),
        "ic_generation_seconds": float(t1_ic_gen - t0_ic_gen) if t0_ic_gen is not None and t1_ic_gen is not None else None,
        "ic_application_seconds": float(t1_ic_apply - t0_ic_apply) if t0_ic_apply is not None and t1_ic_apply is not None else None,
    }

    flops_approx = 0
    for module in getattr(model, "modules", lambda: [])():
        if isinstance(module, nn.Linear):
            flops_approx += 2 * int(getattr(module, "in_features", 0)) * int(getattr(module, "out_features", 0))

    ic_rf_stats = None
    rf_stats_path = ic_rules_dir / "ic_rf_stats.json"
    if rf_stats_path.exists():
        with rf_stats_path.open("r", encoding="utf-8") as f_rf:
            ic_rf_stats = json.load(f_rf)

    summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "splits": {
            "test_size": splits_cfg.get("test_size"),
            "val_size": splits_cfg.get("val_size"),
            "residual_size": splits_cfg.get("residual_size"),
        },
        "split_counts": split_counts,
        "seed": seed,
        "config_snapshot": config,
        "metrics_val": metrics_val,
        "metrics_pre": metrics_pre,
        "metrics_post": metrics_post,
        "baseline_rf": baseline_rf_metrics,
        "ic": {
            "mode": mode,
            "num_rules": len(ic_rules),
            "fraction_flagged": frac_flagged,
        },
        "timing": timing,
        "model_complexity": {
            "nn_num_params": num_params,
            "nn_flops_approx": flops_approx,
            "ic_rf": ic_rf_stats,
        },
    }

    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    history_path = output_dirs["logs"] / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f_hist:
        json.dump(history, f_hist, indent=2)

    return summary


def run_experiment_from_config(config: dict) -> dict:
    """Öffentliche API für run_all_experiments: nimmt Config-Dict, gibt Summary zurück."""
    return _execute_experiment(config)


def run_experiment(config_path: str) -> None:
    """CLI-Entry: lädt YAML von Pfad und führt ein einzelnes Experiment aus."""
    config = load_config(config_path)
    # experiment_name für CLI standardmäßig aus Dateinamen ableiten, falls nicht gesetzt
    config.setdefault("experiment_name", Path(config_path).stem)
    _execute_experiment(config)
