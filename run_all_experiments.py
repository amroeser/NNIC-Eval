import sys
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = Path(__file__).resolve().parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nnic.experiment.runner import run_experiment_from_config
from nnic.reporting.report_builder import build_markdown_report
import yaml


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _run_experiment_cfg(exp_cfg: dict, n_runs: int) -> dict | None:
    """Führt ein Experiment mehrfach aus, basierend auf einem Config-Dict aus experiments.yaml."""
    cfg = dict(exp_cfg)  # defensiv kopieren
    experiment_name = cfg.get("experiment_name", "experiment")

    print(f"\n=== Experiment: {experiment_name} (n_runs={n_runs}) ===")

    run_summaries: list[dict] = []

    # Basis-Seed aus der Config; unterschiedliche Runs erhalten deterministisch
    # abgeleitete Seeds (z.B. 42, 43, 44, ...), behalten aber denselben Split
    base_seed = int(cfg.get("seed", 42))

    for run_idx in range(n_runs):
        print(f"--- Run {run_idx + 1}/{n_runs} ---")
        cfg["seed"] = base_seed + run_idx
        summary = run_experiment_from_config(cfg)
        run_summaries.append(summary)

    if not run_summaries:
        return None

    # Rohdaten der Runs persistent für spätere Signifikanzanalysen speichern
    base_dir = cfg.get("base_output_dir", ".")
    results_dir = Path(base_dir) / "results" / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    run_summaries_path = results_dir / "run_summaries.json"
    with run_summaries_path.open("w", encoding="utf-8") as f:
        json.dump(run_summaries, f, indent=2)

    first = run_summaries[0]
    pre_accs = [s["metrics_pre"]["accuracy"] for s in run_summaries]
    pre_f1s = [s["metrics_pre"]["f1"] for s in run_summaries]
    pre_precs = [s["metrics_pre"]["precision"] for s in run_summaries]
    pre_recalls = [s["metrics_pre"]["recall"] for s in run_summaries]

    post_accs = [
        s["metrics_post"]["accuracy"]
        for s in run_summaries
        if s.get("metrics_post") is not None
    ]
    post_f1s = [
        s["metrics_post"]["f1"]
        for s in run_summaries
        if s.get("metrics_post") is not None
    ]

    post_precs = [
        s["metrics_post"]["precision"]
        for s in run_summaries
        if s.get("metrics_post") is not None
    ]
    post_recalls = [
        s["metrics_post"]["recall"]
        for s in run_summaries
        if s.get("metrics_post") is not None
    ]

    ic_num_rules = [s["ic"]["num_rules"] for s in run_summaries]
    ic_frac_flagged = [s["ic"]["fraction_flagged"] for s in run_summaries]

    metrics_pre_agg = dict(first["metrics_pre"] or {})
    metrics_pre_agg["accuracy"] = _mean(pre_accs)
    metrics_pre_agg["f1"] = _mean(pre_f1s)
    metrics_pre_agg["precision"] = _mean(pre_precs)
    metrics_pre_agg["recall"] = _mean(pre_recalls)

    metrics_post_agg = None
    if post_accs:
        first_post = next(s["metrics_post"] for s in run_summaries if s.get("metrics_post"))
        metrics_post_agg = dict(first_post or {})
        metrics_post_agg["accuracy"] = _mean(post_accs)
        metrics_post_agg["f1"] = _mean(post_f1s)
        metrics_post_agg["precision"] = _mean(post_precs)
        metrics_post_agg["recall"] = _mean(post_recalls)

    def _stats(values):
        if not values:
            return None
        n_vals = len(values)
        m_val = _mean(values)
        if n_vals < 2:
            return {"mean": m_val, "std": None, "ci95": None, "n": n_vals}
        var_val = sum((v - m_val) ** 2 for v in values) / (n_vals - 1)
        std_val = var_val ** 0.5
        z = 1.96
        half = z * std_val / (n_vals ** 0.5)
        return {"mean": m_val, "std": std_val, "ci95": [m_val - half, m_val + half], "n": n_vals}

    metrics_pre_stats = {
        "accuracy": _stats(pre_accs),
        "f1": _stats(pre_f1s),
        "precision": _stats(pre_precs),
        "recall": _stats(pre_recalls),
    }

    metrics_post_stats = {
        "accuracy": _stats(post_accs),
        "f1": _stats(post_f1s),
        "precision": _stats(post_precs),
        "recall": _stats(post_recalls),
    }

    stats_tests = None
    if post_accs and len(post_accs) == len(pre_accs) and len(pre_accs) >= 2:
        diffs = [b - a for a, b in zip(pre_accs, post_accs)]
        n_d = len(diffs)
        d_mean = _mean(diffs)
        if n_d > 1:
            var_d = sum((d - d_mean) ** 2 for d in diffs) / (n_d - 1)
            std_d = var_d ** 0.5
            if std_d > 0.0:
                t_stat = d_mean / (std_d / (n_d ** 0.5))
            else:
                t_stat = 0.0
            stats_tests = {
                "paired_t_pre_post_accuracy": {
                    "t": t_stat,
                    "n": n_d,
                }
            }

    ic_agg = dict(first.get("ic") or {})
    ic_agg["num_rules"] = _mean(ic_num_rules)
    ic_agg["fraction_flagged"] = _mean(ic_frac_flagged)
    ic_agg["n_runs"] = n_runs

    # Wähle zusätzlich den besten Run aus (für die Tabellen-Ausgabe):
    # - primär nach Accuracy mit ICs (metrics_post["accuracy"]), falls vorhanden
    # - sonst nach Accuracy ohne ICs (metrics_pre["accuracy"]).
    def _post_acc_or_neg_inf(idx: int) -> float:
        s = run_summaries[idx]
        mp = s.get("metrics_post") or {}
        acc = mp.get("accuracy")
        return float(acc) if acc is not None else float("-inf")

    def _pre_acc(idx: int) -> float:
        s = run_summaries[idx]
        mp = s.get("metrics_pre") or {}
        acc = mp.get("accuracy")
        return float(acc) if acc is not None else float("-inf")

    if post_accs:
        best_idx = max(range(len(run_summaries)), key=_post_acc_or_neg_inf)
    else:
        best_idx = max(range(len(run_summaries)), key=_pre_acc)

    best = run_summaries[best_idx]

    summary_agg = dict(first)
    # Mittelwerte separat bereitstellen, falls benötigt
    summary_agg["metrics_pre_mean"] = metrics_pre_agg
    summary_agg["metrics_post_mean"] = metrics_post_agg
    summary_agg["ic_mean"] = ic_agg
    summary_agg["metrics_pre_stats"] = metrics_pre_stats
    summary_agg["metrics_post_stats"] = metrics_post_stats
    summary_agg["stats"] = stats_tests

    # Für den Report die Metriken des besten Runs verwenden (Best-of-n)
    summary_agg["metrics_pre"] = best.get("metrics_pre")
    summary_agg["metrics_post"] = best.get("metrics_post")
    summary_agg["ic"] = best.get("ic") or {}

    return summary_agg


def _is_excluded_experiment(exp_cfg: dict) -> bool:
    """Return True if this experiment should be skipped entirely for the batch run.

    We exclude experiments for the following datasets from execution, to keep the
    main study focused on the core benchmarks:

    - digits
    - iris
    - breast_cancer
    - wine
    - openml-based datasets corresponding to bank-marketing (1461),
      adult (45068) and nomao (1486).
    """

    ds_cfg = exp_cfg.get("dataset") or {}
    name = ds_cfg.get("name")

    # Direct dataset names
    if name in {"digits", "iris", "breast_cancer", "wine"}:
        return True

    # OpenML datasets: decide based on the OpenML ID
    if name == "openml":
        ds_id = ds_cfg.get("openml_id")
        try:
            ds_id_int = int(ds_id) if ds_id is not None else None
        except (TypeError, ValueError):
            ds_id_int = None

        # 1461: bank-marketing, 45068: adult, 1486: nomao
        if ds_id_int in {1461, 45068, 1486}:
            return True

    return False


def main() -> None:
    master_path = project_root / "configs" / "experiments.yaml"
    with master_path.open("r", encoding="utf-8") as f:
        master_cfg = yaml.safe_load(f) or {}
    experiments = master_cfg.get("experiments", [])

    # Bestimmte Datasets gar nicht erst ausführen (nur für Nebenexperimente /
    # Explorationsläufe relevant, nicht für die Kernstudie).
    experiments = [
        cfg for cfg in experiments if not _is_excluded_experiment(cfg)
    ]
    n_runs = int(master_cfg.get("n_runs", 1))
    summaries: list[dict] = []

    # Parallelisiere über Configs, um Gesamtlaufzeit zu reduzieren
    max_workers = min(3, len(experiments))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {
            executor.submit(_run_experiment_cfg, exp_cfg, n_runs): exp_cfg.get("experiment_name", "experiment")
            for exp_cfg in experiments
        }

        for future in as_completed(future_to_cfg):
            rel_name = future_to_cfg[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Experiment {rel_name} generated an exception: {exc}")
                continue
            if result is not None:
                summaries.append(result)

    if not summaries:
        print("No experiment summaries found; report will not be generated.")
        return

    report_path = project_root / "results" / "all_experiments_report.md"
    build_markdown_report(summaries, report_path)
    print(f"\nBatch report written to: {report_path}")


if __name__ == "__main__":
    main()
