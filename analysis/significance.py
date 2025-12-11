"""Significance analysis for NNIC experiments.

This script reads per-experiment run_summaries.json files created by
run_all_experiments.py (one JSON list per experiment containing the raw
per-run summaries), computes paired tests between Acc pre and Acc post,
and writes a Markdown report with basic statistics and p-values.

Usage (from project root):

    python -m analysis.significance

Requires numpy and optionally SciPy. If SciPy is not installed, the script
will still compute means and deltas, but p-values will be reported as '-'.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.stats import ttest_rel, wilcoxon  # type: ignore[import]

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    ttest_rel = None  # type: ignore[assignment]
    wilcoxon = None  # type: ignore[assignment]
    _HAS_SCIPY = False

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore[import]

    _HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False


def _load_run_summaries(results_dir: Path) -> List[Dict]:
    run_file = results_dir / "run_summaries.json"
    if not run_file.exists():
        return []
    with run_file.open("r", encoding="utf-8") as f:
        data = json.load(f) or []
    assert isinstance(data, list)
    return data  # type: ignore[return-value]


def _extract_pre_post_pairs(run_summaries: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    pre_vals: List[float] = []
    post_vals: List[float] = []
    for s in run_summaries:
        pre = (s.get("metrics_pre") or {}).get("accuracy")
        post = (s.get("metrics_post") or {}).get("accuracy") if s.get("metrics_post") else None
        if pre is None or post is None:
            continue
        pre_vals.append(float(pre))
        post_vals.append(float(post))
    return np.asarray(pre_vals, dtype=float), np.asarray(post_vals, dtype=float)


def _bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int = 42) -> Tuple[float, float]:
    """Return percentile bootstrap 95% CI for the mean of diffs.

    Uses simple i.i.d. resampling with replacement. For small n this mainly
    provides a robustness check on the estimated mean difference.
    """

    if diffs.size == 0:
        raise ValueError("diffs must be non-empty for bootstrap")

    rng = np.random.default_rng(seed)
    # Shape: (n_boot, n_samples)
    samples = rng.choice(diffs, size=(n_boot, diffs.size), replace=True)
    boot_means = samples.mean(axis=1)
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return low, high


def _summarise_experiment(exp_name: str, results_dir: Path) -> Dict:
    run_summaries = _load_run_summaries(results_dir)
    pre, post = _extract_pre_post_pairs(run_summaries)

    if pre.size == 0 or post.size == 0:
        return {
            "experiment": exp_name,
            "n_runs_total": len(run_summaries),
            "n_runs_with_ic": 0,
            "mean_pre": None,
            "mean_post": None,
            "mean_delta": None,
            "mean_enh": None,
            "std_delta": None,
            "ci_low": None,
            "ci_high": None,
            "boot_ci_1000_low": None,
            "boot_ci_1000_high": None,
            "boot_ci_2000_low": None,
            "boot_ci_2000_high": None,
            "t_stat": None,
            "p_value": None,
            "w_stat": None,
            "p_value_wilcoxon": None,
            "diffs": [],
        }

    assert pre.shape == post.shape
    diffs = post - pre

    mean_pre = float(pre.mean())
    mean_post = float(post.mean())
    mean_delta = float(diffs.mean())
    mean_enh = float(100.0 * mean_delta / mean_pre) if mean_pre > 0 else None

    std_delta = None
    ci_low = None
    ci_high = None
    boot_ci_1000_low = None
    boot_ci_1000_high = None
    boot_ci_2000_low = None
    boot_ci_2000_high = None
    if diffs.size > 1:
        std_delta = float(diffs.std(ddof=1))
        half_width = 1.96 * std_delta / sqrt(float(diffs.size))
        ci_low = float(mean_delta - half_width)
        ci_high = float(mean_delta + half_width)

        # Bootstrap-Konfidenzintervalle für die mittlere Differenz, um die
        # Stabilität der Schätzung bei kleiner Stichprobe abzuschätzen.
        boot_ci_1000_low, boot_ci_1000_high = _bootstrap_ci(diffs, 1000)
        boot_ci_2000_low, boot_ci_2000_high = _bootstrap_ci(diffs, 2000)

    t_stat = None
    p_value = None
    w_stat = None
    p_value_wilcoxon = None
    if _HAS_SCIPY and pre.size > 1:
        t_res = ttest_rel(post, pre)
        t_stat = float(t_res.statistic)
        p_value = float(t_res.pvalue)

        try:
            w_res = wilcoxon(post, pre)  # type: ignore[operator]
            w_stat = float(w_res.statistic)
            p_value_wilcoxon = float(w_res.pvalue)
        except Exception:
            w_stat = None
            p_value_wilcoxon = None

    return {
        "experiment": exp_name,
        "n_runs_total": len(run_summaries),
        "n_runs_with_ic": int(pre.size),
        "mean_pre": mean_pre,
        "mean_post": mean_post,
        "mean_delta": mean_delta,
        "mean_enh": mean_enh,
        "std_delta": std_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "boot_ci_1000_low": boot_ci_1000_low,
        "boot_ci_1000_high": boot_ci_1000_high,
        "boot_ci_2000_low": boot_ci_2000_low,
        "boot_ci_2000_high": boot_ci_2000_high,
        "t_stat": t_stat,
        "p_value": p_value,
        "w_stat": w_stat,
        "p_value_wilcoxon": p_value_wilcoxon,
        "diffs": diffs.tolist(),
    }


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _fmt_interval(low, high) -> str:
    if low is None or high is None:
        return "-"
    return f"[{low:.3f}, {high:.3f}]"


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_root = project_root / "results"
    if not results_root.exists():
        print(f"No results directory found at {results_root}")
        return

    # Datensätze, die nicht Teil der Kernstudie sind, aus der
    # Signifikanzanalyse ausschließen, analog zum Hauptreport.
    excluded_datasets = {
        "adult",
        "bank-marketing",
        "breast_cancer",
        "digits",
        "iris",
        "nomao",
        "wine",
    }

    rows: List[Dict] = []
    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        run_file = exp_dir / "run_summaries.json"
        if not run_file.exists():
            continue

        # Dataset-Name aus summary.json lesen (falls vorhanden), um auf
        # Dataset-Ebene filtern zu können.
        summary_file = exp_dir / "summary.json"
        dataset_name = None
        if summary_file.exists():
            try:
                with summary_file.open("r", encoding="utf-8") as f:
                    summary_data = json.load(f) or {}
                dataset_name = summary_data.get("dataset")
            except Exception:
                dataset_name = None

        if dataset_name in excluded_datasets:
            continue

        rows.append(_summarise_experiment(exp_dir.name, exp_dir))

    if not rows:
        print("No run_summaries.json files found; did you run run_all_experiments with n_runs > 1?")
        return

    report_path = results_root / "significance_report.md"
    lines: List[str] = []
    lines.append("# NNIC Significance Report")
    lines.append("")
    lines.append("This report summarises paired comparisons between test accuracy before (Acc pre) "
                 "and after IC application (Acc post) across multiple runs for each experiment.")
    lines.append("")
    lines.append("Note that n_runs is relatively small for most experiments; therefore, classical "
                 "significance tests have limited power. We primarily interpret the direction and "
                 "magnitude of the mean accuracy change (ΔAcc) and its confidence interval.")
    lines.append("")
    if not _HAS_SCIPY:
        lines.append(
            "> SciPy is not installed; p-values are not computed. Install `scipy` to enable t-tests."
        )
        lines.append("")

    lines.append("| Experiment | n runs (IC) | Acc pre (mean) | Acc post (mean) | ΔAcc (mean) | Enh% (mean) | ΔAcc std | 95% CI low | 95% CI high | Boot 95% CI (B=1000) | Boot 95% CI (B=2000) | t-stat | p (t-test) | W-stat | p (Wilcoxon) |")
    lines.append("|-----------|-------------:|---------------:|----------------:|------------:|------------:|---------:|-----------:|------------:|-----------------------:|-----------------------:|-------:|-----------:|-------:|-------------:|")

    # Nur Experimente mit mindestens 2 Runs mit ICs in der Tabelle anzeigen,
    # da für n_runs_with_ic <= 1 keine sinnvollen Streuungsmaße bzw.
    # Signifikanztests berechnet werden können.
    filtered_rows = [r for r in rows if r["n_runs_with_ic"] >= 2]

    # Optionale Visualisierung: Eine zusammenfassende Grafik ("Forest Plot"),
    # die für jedes Experiment den mittleren ΔAcc mit Bootstrap-95%-CI zeigt.
    if _HAS_MPL and filtered_rows:
        plots_dir = results_root / "plots" / "significance"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Für die Visualisierung nach Effektgröße sortieren (absteigend).
        sorted_rows = sorted(
            filtered_rows,
            key=lambda d: d.get("mean_delta") or 0.0,
            reverse=True,
        )
        n_exp = len(sorted_rows)

        # Mittelwerte und Bootstrap-CIs (B=2000) extrahieren
        means = []
        ci_lows = []
        ci_highs = []
        labels_exp = []
        for r in sorted_rows:
            m = r.get("mean_delta")
            low = r.get("boot_ci_2000_low") or r.get("ci_low")
            high = r.get("boot_ci_2000_high") or r.get("ci_high")
            if m is None or low is None or high is None:
                continue
            means.append(float(m))
            ci_lows.append(float(low))
            ci_highs.append(float(high))
            labels_exp.append(r["experiment"])

        if means:
            # Forest-Plot der Mittelwerte mit Bootstrap-CIs
            y_pos = np.arange(len(means))
            fig, ax = plt.subplots(figsize=(8, 0.6 * len(means) + 1.5))

            for i, (m, low, high) in enumerate(zip(means, ci_lows, ci_highs)):
                err_left = m - low
                err_right = high - m
                ax.errorbar(
                    m,
                    i,
                    xerr=[[err_left], [err_right]],
                    fmt="o",
                    capsize=3,
                    color="C0",
                )

            ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_exp, fontsize=8)
            ax.set_xlabel("ΔAcc (Acc post - Acc pre)")
            ax.set_title("Mean ΔAcc with bootstrap 95% CIs (B=2000)")
            fig.tight_layout()
            fig_path = plots_dir / "delta_acc_forest.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)

    for r in sorted(filtered_rows, key=lambda d: d["experiment"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    r["experiment"],
                    _fmt(r["n_runs_with_ic"]),
                    _fmt(r["mean_pre"]),
                    _fmt(r["mean_post"]),
                    _fmt(r["mean_delta"]),
                    _fmt(r["mean_enh"]),
                    _fmt(r["std_delta"]),
                    _fmt(r["ci_low"]),
                    _fmt(r["ci_high"]),
                    _fmt_interval(r["boot_ci_1000_low"], r["boot_ci_1000_high"]),
                    _fmt_interval(r["boot_ci_2000_low"], r["boot_ci_2000_high"]),
                    _fmt(r["t_stat"]),
                    _fmt(r["p_value"]),
                    _fmt(r["w_stat"]),
                    _fmt(r["p_value_wilcoxon"]),
                ]
            )
            + " |"
        )

    # Falls die zusammenfassende Abbildung erzeugt wurde, im Report verlinken.
    if _HAS_MPL and filtered_rows:
        lines.append("")
        lines.append("## Summary visualisation of ΔAcc")
        lines.append("")
        lines.append(
            "![Per-experiment mean ΔAcc with bootstrap 95% CIs](plots/significance/delta_acc_forest.png)"
        )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Significance report written to: {report_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
