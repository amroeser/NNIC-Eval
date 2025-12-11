from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt  # type: ignore[import]

    _HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False


def _split_percents(splits: Dict[str, Any]) -> Dict[str, float]:
    test = float(splits.get("test_size", 0.0) or 0.0)
    val = float(splits.get("val_size", 0.0) or 0.0)
    residual = float(splits.get("residual_size", 0.0) or 0.0)
    train = max(0.0, 1.0 - test - val - residual)

    def pct(x: float) -> float:
        return round(100 * x, 1)

    return {
        "train": pct(train),
        "residual": pct(residual),
        "val": pct(val),
        "test": pct(test),
    }


def build_markdown_report(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# NNIC Batch Experiments Report")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("- **Acc pre / Acc post**: Test accuracy of the NN without vs. with NNIC (IC corrections).")
    lines.append("- **ΔAcc**: Absolute difference `Acc post - Acc pre`.")
    lines.append(
        "- **Enh%**: Relative improvement in test accuracy due to NNIC, "
        "`100 * (Acc post - Acc pre) / Acc pre` (only defined if `Acc pre > 0` and ICs are applied)."
    )
    lines.append("- **F1 pre / F1 post**: F1-score before vs. after applying ICs.")
    lines.append("- **#IC rules**: Number of generated and applied IC rules.")
    lines.append(
        "- **frac flagged**: Fraction of test examples for which at least one IC fired "
        "and changed the NN prediction (flagging/correction rate)."
    )
    lines.append(
        "- **Err pre / Err post / ErrRed%**: Number of errors before vs. after applying ICs "
        "and the relative error reduction, `100 * (Err pre - Err post) / Err pre`."
    )
    lines.append(
        "- **Acc/F1 RF (optional)**: Accuracy and F1-score of a random-forest baseline classifier "
        "(if configured)."
    )
    lines.append("")

    # Nur ausgewählte Datensätze in die Studie aufnehmen
    excluded_datasets = {
        "adult",
        "bank-marketing",
        "breast_cancer",
        "digits",
        "iris",
        "nomao",
        "wine",
    }

    filtered_summaries = [
        s for s in summaries if s.get("dataset") not in excluded_datasets
    ]

    lines.append("## Green-AI (potential benefit)")
    lines.append("")
    lines.append(
        "The following experiments record, among other things, the number of NN parameters "
        "as well as training and IC runtimes. These quantities are indicators of resource "
        "consumption but **are not direct energy measurements**. Any Green-AI statements "
        "about NNIC therefore refer to a **potential** reduction in compute effort and "
        "error rates, not to measured CO₂ emissions."
    )
    lines.append("")

    # Globale, testfall-gewichtete Wirkung von NNIC über alle Experimente mit ICs
    total_test = 0
    sum_acc_pre = 0.0
    sum_acc_post = 0.0
    for s in filtered_summaries:
        pre = s.get("metrics_pre") or {}
        post = s.get("metrics_post") or {}
        acc_pre = pre.get("accuracy")
        acc_post = post.get("accuracy")
        if acc_pre is None or acc_post is None:
            continue
        counts = s.get("split_counts", {})
        n_test = int(counts.get("test", 0) or 0)
        if n_test <= 0:
            continue
        total_test += n_test
        sum_acc_pre += float(acc_pre) * n_test
        sum_acc_post += float(acc_post) * n_test

    if total_test > 0 and sum_acc_pre > 0.0:
        acc_pre_w = sum_acc_pre / total_test
        acc_post_w = sum_acc_post / total_test
        delta_w = acc_post_w - acc_pre_w
        enh_w = 100.0 * delta_w / acc_pre_w if acc_pre_w > 0.0 else None

        lines.append("### Global effect of NNIC (weighted means over all experiments with ICs)")
        lines.append("")
        lines.append(f"- **Acc pre (weighted, only experiments with ICs)**: {acc_pre_w:.3f}")
        lines.append(f"- **Acc post (weighted, only experiments with ICs)**: {acc_post_w:.3f}")
        lines.append(f"- **ΔAcc (weighted)**: {delta_w:.3f}")
        if enh_w is not None:
            lines.append(f"- **Enh% (weighted)**: {enh_w:.2f}%")
        lines.append(f"- **Number of test examples (only experiments with ICs)**: {total_test}")
        lines.append("")

        # Vergleich: NNIC mit halber Trainingsbasis vs. 95%-Baseline (Mittelwerte über Runs)
        #  - NNIC: 47.5% Train, 47.5% Residual, 5% Test -> Acc post (mean)
        #  - Baseline: 95% Train, 0% Residual, 5% Test -> Acc pre (mean)

        comparison: Dict[str, Dict[str, float]] = {}

        for s in filtered_summaries:
            dataset_name = s.get("dataset", "unknown")
            splits = s.get("splits", {})
            sp = _split_percents(splits)

            train_pct = float(sp.get("train", 0.0) or 0.0)
            residual_pct = float(sp.get("residual", 0.0) or 0.0)

            pre_mean = (s.get("metrics_pre_mean") or {}).get("accuracy")
            post_mean = (s.get("metrics_post_mean") or {}).get("accuracy")

            if dataset_name not in comparison:
                comparison[dataset_name] = {}

            # 95% Train / 0% Residual / 5% Test: Baseline ohne ICs
            if abs(train_pct - 95.0) < 1e-6 and abs(residual_pct - 0.0) < 1e-6:
                if pre_mean is not None:
                    comparison[dataset_name]["baseline_pre"] = float(pre_mean)

            # 47.5% Train / 47.5% Residual / 5% Test: NNIC mit halber Trainingsbasis
            if abs(train_pct - 47.5) < 1e-6 and abs(residual_pct - 47.5) < 1e-6:
                if post_mean is not None:
                    comparison[dataset_name]["nnic_post"] = float(post_mean)

        # Nur Datensätze berücksichtigen, die beide Konfigurationen besitzen
        ordered_datasets = [
            "covertype",
            "credit-g",
            "electricity",
            "magic-telescope",
            "phoneme",
            "spambase",
        ]

        rows: List[str] = []
        for ds in ordered_datasets:
            entry = comparison.get(ds) or {}
            baseline = entry.get("baseline_pre")
            nnic = entry.get("nnic_post")
            if baseline is None or nnic is None:
                continue

            delta = nnic - baseline
            rel = (100.0 * delta / baseline) if baseline > 0.0 else None

            baseline_str = f"{baseline:.3f}"
            nnic_str = f"{nnic:.3f}"
            delta_str = f"{delta:+.3f}"
            rel_str = f"{rel:+.1f}%" if rel is not None else "-"

            rows.append(
                f"| {ds:16} | {baseline_str:>38} | {nnic_str:>32} | {delta_str:>22} | {rel_str:>20} |"
            )

        if rows:
            lines.append(
                "### Comparison: NNIC with half training data vs. 95% NN baseline (mean over runs)"
            )
            lines.append("")
            lines.append(
                "The following overview compares, for selected datasets, the performance of NNIC with half the "
                "training data (47.5% train, 47.5% residual, 5% test; metric: `Acc post (mean)`) to the plain NN "
                "baseline with 95% training data (95% train, 0% residual, 5% test; metric: `Acc pre (mean)`)."
            )
            lines.append("")
            lines.append(
                "| Dataset          | NN baseline (95% train, Acc pre mean) | NNIC (47.5/47.5, Acc post mean) | ΔAcc (NNIC - 95% NN) | Rel. ΔAcc vs. 95% NN |"
            )
            lines.append(
                "|------------------|----------------------------------------|----------------------------------|----------------------|----------------------|"
            )
            lines.extend(rows)
            lines.append("")

    datasets = sorted({s.get("dataset", "unknown") for s in filtered_summaries})

    # Collect all plots in a common folder
    plots_root = output_path.parent.parent / "plots" / "report"
    if _HAS_MPL:
        plots_root.mkdir(parents=True, exist_ok=True)

        # Global overview: IC rules vs. accuracy improvement across all experiments
        global_rules: List[float] = []
        global_enh: List[float] = []

        for s in filtered_summaries:
            pre = s.get("metrics_pre") or {}
            post = s.get("metrics_post") or {}
            acc_pre = pre.get("accuracy")
            acc_post = post.get("accuracy")
            ic = s.get("ic") or {}
            num_rules = ic.get("num_rules")

            if (
                acc_pre is not None
                and acc_post is not None
                and acc_pre > 0.0
                and num_rules is not None
            ):
                delta = float(acc_post) - float(acc_pre)
                enh = 100.0 * delta / float(acc_pre)
                global_rules.append(float(num_rules))
                global_enh.append(float(enh))

        if global_rules and global_enh:
            try:
                fig_all, ax_all = plt.subplots(figsize=(6.5, 4.0))  # type: ignore[call-arg]
                ax_all.scatter(global_rules, global_enh, color="#c44e52")
                ax_all.set_xlabel("#IC rules", fontsize=10)
                ax_all.set_ylabel("Enh% (relative accuracy improvement)", fontsize=10)
                fig_all.tight_layout()

                plot_all_path = plots_root / "all_experiments_ic_effect.png"
                fig_all.savefig(plot_all_path, dpi=200)  # type: ignore[arg-type]
                plt.close(fig_all)  # type: ignore[call-arg]

                rel_all_plot_path = "../plots/report/all_experiments_ic_effect.png"
                lines.append("### Overview: IC rules vs. accuracy improvement (all experiments)")
                lines.append("")
                lines.append(
                    f"![IC rules vs. accuracy improvement – all experiments]({rel_all_plot_path})"
                )
                lines.append("")
            except Exception:
                # If global plot creation fails, silently continue with the report.
                pass

    for dataset in datasets:
        ds_summaries = [s for s in filtered_summaries if s.get("dataset") == dataset]
        if not ds_summaries:
            continue
        lines.append(f"## Dataset: {dataset}")
        lines.append("")
        lines.append("| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre | Acc post | ΔAcc | Enh% | Err pre | Err post | ErrRed% | F1 pre | F1 post | #IC rules | frac flagged | Acc RF | F1 RF |")
        lines.append("|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------:|---------:|-----:|------:|--------:|---------:|--------:|-------:|--------:|----------:|-------------:|-------:|------:|")

        # Für optionale Visualisierung sammeln wir pro Dataset die Akkuratheitswerte
        plot_labels: List[str] = []  # für x-Tick-Labels (Train%)
        plot_pre: List[float] = []   # Acc pre je Split
        plot_post: List[float] = []  # Acc post je Split
        plot_x: List[float] = []     # numerische Train%-Werte für die x-Achse
        plot_x_err: List[float] = []
        plot_err_pre: List[float] = []
        plot_err_post: List[float] = []
        plot_rules: List[float] = []
        plot_enh: List[float] = []

        for s in sorted(ds_summaries, key=lambda x: x.get("experiment_name", "")):
            splits = s.get("splits", {})
            sp = _split_percents(splits)
            counts = s.get("split_counts", {})
            n_train = counts.get("train", 0)
            n_residual = counts.get("residual", 0)
            n_test = counts.get("test", 0)
            name = s.get("experiment_name", "")
            pre = s.get("metrics_pre") or {}
            post = s.get("metrics_post") or {}
            acc_pre = pre.get("accuracy")
            acc_post = post.get("accuracy") if post else None
            f1_pre = pre.get("f1")
            f1_post = post.get("f1") if post else None
            err_pre = pre.get("num_errors")
            err_post = post.get("num_errors") if post else None
            delta_acc = (acc_post - acc_pre) if (acc_post is not None and acc_pre is not None) else None
            enhancement = None
            if delta_acc is not None and acc_pre and acc_pre > 0.0:
                enhancement = 100.0 * delta_acc / acc_pre

            err_reduction_pct = None
            if err_pre is not None and err_post is not None and err_pre > 0:
                err_reduction_pct = 100.0 * (float(err_pre) - float(err_post)) / float(err_pre)

            ic = s.get("ic") or {}
            num_rules = ic.get("num_rules", 0)
            frac_flagged = ic.get("fraction_flagged", 0.0)

            if enhancement is not None:
                plot_rules.append(float(num_rules))
                plot_enh.append(float(enhancement))

            baseline_rf = s.get("baseline_rf") or {}
            acc_rf = baseline_rf.get("accuracy")
            f1_rf = baseline_rf.get("f1")

            if acc_pre is not None and acc_post is not None:
                train_pct = float(sp["train"])
                plot_x.append(train_pct)
                plot_labels.append(f"{sp['train']}")
                plot_pre.append(float(acc_pre))
                plot_post.append(float(acc_post))

            if err_pre is not None and err_post is not None:
                train_pct_err = float(sp["train"])
                plot_x_err.append(train_pct_err)
                plot_err_pre.append(float(err_pre))
                plot_err_post.append(float(err_post))

            def fmt(x: Any) -> str:
                if x is None:
                    return "-"
                if isinstance(x, float):
                    return f"{x:.3f}"
                return str(x)

            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt(sp["train"]),
                        fmt(sp["residual"]),
                        fmt(sp["test"]),
                        fmt(n_train),
                        fmt(n_residual),
                        fmt(n_test),
                        name,
                        fmt(acc_pre),
                        fmt(acc_post),
                        fmt(delta_acc),
                        fmt(enhancement),
                        fmt(err_pre),
                        fmt(err_post),
                        fmt(err_reduction_pct),
                        fmt(f1_pre),
                        fmt(f1_post),
                        fmt(num_rules),
                        fmt(frac_flagged),
                        fmt(acc_rf),
                        fmt(f1_rf),
                    ]
                )
                + " |"
            )

        lines.append("")

        # Falls mehrere Runs existieren, optionale zweite Tabelle mit Mittelwerten anzeigen
        ds_summaries_with_mean = [s for s in ds_summaries if s.get("metrics_pre_mean")]
        if ds_summaries_with_mean:
            lines.append("### Mittelwerte über Runs (sofern n_runs > 1)")
            lines.append("")
            lines.append("| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre (mean) | Acc post (mean) | ΔAcc (mean) | Enh% (mean) | F1 pre (mean) | F1 post (mean) | #IC rules (mean) | frac flagged (mean) |")
            lines.append("|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------------:|----------------:|------------:|------------:|--------------:|---------------:|------------------:|--------------------:|")

            for s in sorted(ds_summaries_with_mean, key=lambda x: x.get("experiment_name", "")):
                splits = s.get("splits", {})
                sp = _split_percents(splits)
                counts = s.get("split_counts", {})
                n_train = counts.get("train", 0)
                n_residual = counts.get("residual", 0)
                n_test = counts.get("test", 0)
                name = s.get("experiment_name", "")
                pre_m = s.get("metrics_pre_mean") or {}
                post_m = s.get("metrics_post_mean") or {}
                acc_pre_m = pre_m.get("accuracy")
                acc_post_m = post_m.get("accuracy") if post_m else None
                f1_pre_m = pre_m.get("f1")
                f1_post_m = post_m.get("f1") if post_m else None
                delta_acc_m = (
                    acc_post_m - acc_pre_m
                    if (acc_post_m is not None and acc_pre_m is not None)
                    else None
                )
                enh_m = None
                if delta_acc_m is not None and acc_pre_m and acc_pre_m > 0.0:
                    enh_m = 100.0 * delta_acc_m / acc_pre_m
                ic_m = s.get("ic_mean") or {}
                num_rules_m = ic_m.get("num_rules", 0.0)
                frac_flagged_m = ic_m.get("fraction_flagged", 0.0)

                def fmt_mean(x: Any) -> str:
                    if x is None:
                        return "-"
                    if isinstance(x, float):
                        return f"{x:.3f}"
                    return str(x)

                lines.append(
                    "| "
                    + " | ".join(
                        [
                            fmt_mean(sp["train"]),
                            fmt_mean(sp["residual"]),
                            fmt_mean(sp["test"]),
                            fmt_mean(n_train),
                            fmt_mean(n_residual),
                            fmt_mean(n_test),
                            name,
                            fmt_mean(acc_pre_m),
                            fmt_mean(acc_post_m),
                            fmt_mean(delta_acc_m),
                            fmt_mean(enh_m),
                            fmt_mean(f1_pre_m),
                            fmt_mean(f1_post_m),
                            fmt_mean(num_rules_m),
                            fmt_mean(frac_flagged_m),
                        ]
                    )
                    + " |"
                )

            lines.append("")

        # Optional: Linienplots Acc pre/post pro Dataset mit akademischem Stil erzeugen
        if _HAS_MPL and plot_x and plot_pre and plot_post:
            try:
                # Nach Train% sortieren, damit die Linie monoton in x verläuft
                order = sorted(range(len(plot_x)), key=lambda i: plot_x[i])
                xs = [plot_x[i] for i in order]
                ys_pre = [plot_pre[i] for i in order]
                ys_post = [plot_post[i] for i in order]
                labels_sorted = [plot_labels[i] for i in order]

                fig, ax = plt.subplots(figsize=(6.5, 4.0))  # type: ignore[call-arg]

                # Dezente, publikationsnahe Farbwahl
                ax.plot(
                    xs,
                    ys_pre,
                    marker="o",
                    color="#4c72b0",
                    label="Acc pre",
                    linewidth=1.5,
                )
                ax.plot(
                    xs,
                    ys_post,
                    marker="s",
                    color="#55a868",
                    label="Acc post",
                    linewidth=1.5,
                )

                ax.set_xticks(xs)
                ax.set_xticklabels(labels_sorted)
                ax.set_xlabel("Train%", fontsize=10)
                ax.set_ylabel("Accuracy", fontsize=10)
                ax.set_ylim(0.0, 1.0)

                # Dezentes Gitter nur auf der y-Achse, obere/rechte Achse entfernen
                ax.yaxis.grid(True, linestyle="--", alpha=0.5)
                ax.xaxis.grid(False)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)

                ax.set_title(f"NN vs. NNIC accuracy – {dataset}", fontsize=11)
                ax.legend(frameon=False, fontsize=9)
                fig.tight_layout()

                plot_path = plots_root / f"{dataset}_acc.png"
                fig.savefig(plot_path, dpi=200)  # type: ignore[arg-type]
                plt.close(fig)  # type: ignore[call-arg]

                rel_plot_path = f"../plots/report/{dataset}_acc.png"
                lines.append(
                    f"![Accuracy pre/post with NNIC for {dataset}]({rel_plot_path})"
                )
                lines.append("")
            except Exception:
                # Wenn Plot-Erzeugung fehlschlägt, ignorieren wir das still und fahren mit dem Report fort.
                pass

        if _HAS_MPL and plot_x_err and plot_err_pre and plot_err_post:
            try:
                order_err = sorted(range(len(plot_x_err)), key=lambda i: plot_x_err[i])
                xs_err = [plot_x_err[i] for i in order_err]
                ys_err_pre = [plot_err_pre[i] for i in order_err]
                ys_err_post = [plot_err_post[i] for i in order_err]

                fig_err, ax_err = plt.subplots(figsize=(6.5, 4.0))  # type: ignore[call-arg]
                indices = list(range(len(xs_err)))
                width = 0.35
                ax_err.bar([x - width / 2 for x in indices], ys_err_pre, width=width, color="#4c72b0", label="Errors pre")
                ax_err.bar([x + width / 2 for x in indices], ys_err_post, width=width, color="#55a868", label="Errors post")
                ax_err.set_xticks(indices)
                ax_err.set_xticklabels([str(int(x)) for x in xs_err])
                ax_err.set_xlabel("Train%", fontsize=10)
                ax_err.set_ylabel("Number of errors", fontsize=10)
                fig_err.tight_layout()

                plot_err_path = plots_root / f"{dataset}_errors.png"
                fig_err.savefig(plot_err_path, dpi=200)  # type: ignore[arg-type]
                plt.close(fig_err)  # type: ignore[call-arg]

                rel_err_plot_path = f"../plots/report/{dataset}_errors.png"
                lines.append(
                    f"![Errors before/after NNIC for {dataset}]({rel_err_plot_path})"
                )
                lines.append("")
            except Exception:
                pass

        if _HAS_MPL and plot_rules and plot_enh:
            try:
                fig_ic, ax_ic = plt.subplots(figsize=(6.5, 4.0))  # type: ignore[call-arg]
                ax_ic.scatter(plot_rules, plot_enh, color="#c44e52")
                ax_ic.set_xlabel("#IC rules", fontsize=10)
                ax_ic.set_ylabel("Enh% (relative accuracy improvement)", fontsize=10)
                fig_ic.tight_layout()

                plot_ic_path = plots_root / f"{dataset}_ic_effect.png"
                fig_ic.savefig(plot_ic_path, dpi=200)  # type: ignore[arg-type]
                plt.close(fig_ic)  # type: ignore[call-arg]

                rel_ic_plot_path = f"../plots/report/{dataset}_ic_effect.png"
                lines.append(
                    f"![Effect of IC rules for {dataset}]({rel_ic_plot_path})"
                )
                lines.append("")
            except Exception:
                pass

        if _HAS_MPL and plot_x_err and plot_err_pre and plot_err_post:
            try:
                order_err = sorted(range(len(plot_x_err)), key=lambda i: plot_x_err[i])
                xs_err = [plot_x_err[i] for i in order_err]
                ys_err_pre = [plot_err_pre[i] for i in order_err]
                ys_err_post = [plot_err_post[i] for i in order_err]

                fig_err, ax_err = plt.subplots(figsize=(6.5, 4.0))  # type: ignore[call-arg]
                indices = list(range(len(xs_err)))
                width = 0.35
                ax_err.bar([x - width / 2 for x in indices], ys_err_pre, width=width, color="#4c72b0", label="Errors pre")
                ax_err.bar([x + width / 2 for x in indices], ys_err_post, width=width, color="#55a868", label="Errors post")
                ax_err.set_xticks(indices)
                ax_err.set_xticklabels([str(int(x)) for x in xs_err])
                ax_err.set_xlabel("Train%", fontsize=10)
                ax_err.set_ylabel("Number of errors", fontsize=10)
                fig_err.tight_layout()

                plot_err_path = plots_root / f"{dataset}_errors.png"
                fig_err.savefig(plot_err_path, dpi=200)  # type: ignore[arg-type]
                plt.close(fig_err)  # type: ignore[call-arg]

                rel_err_plot_path = f"../plots/report/{dataset}_errors.png"
                lines.append(f"![Fehler vor/nach NNIC für {dataset}]({rel_err_plot_path})")
                lines.append("")
            except Exception:
                pass

        if _HAS_MPL and plot_rules and plot_enh:
            try:
                fig_ic, ax_ic = plt.subplots(figsize=(6.5, 4.0))  # type: ignore[call-arg]
                ax_ic.scatter(plot_rules, plot_enh, color="#c44e52")
                ax_ic.set_xlabel("#IC rules", fontsize=10)
                ax_ic.set_ylabel("Enh%", fontsize=10)
                fig_ic.tight_layout()

                plot_ic_path = plots_root / f"{dataset}_ic_effect.png"
                fig_ic.savefig(plot_ic_path, dpi=200)  # type: ignore[arg-type]
                plt.close(fig_ic)  # type: ignore[call-arg]

                rel_ic_plot_path = f"../plots/report/{dataset}_ic_effect.png"
                lines.append(f"![Effekt der IC-Regeln für {dataset}]({rel_ic_plot_path})")
                lines.append("")
            except Exception:
                pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
