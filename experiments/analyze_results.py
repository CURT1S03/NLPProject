"""
analyze_results.py
Loads results/results.csv and produces per-dataset and cross-dataset plots.

Plots produced per dataset:
  Plot 1  — accuracy vs. severity (2×2 grid)
  Plot 2  — accuracy drop heatmap (atomic conditions)
  Plot 3  — combination perturbations heatmap
  Plot 4  — per-class accuracy (positive vs. negative)
  Plot 5  — flip-rate vs. severity (2×2 grid)

Plots produced once (cross-dataset):
  Plot 6  — DistilBERT confidence curves
  Plot 7  — dataset comparison grouped bar
  Plot 8  — McNemar significance matrix (if mcnemar_results.csv exists)

Usage:
  python -m experiments.analyze_results
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from data.perturbations import SEVERITIES

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
_CSV         = os.path.join(_RESULTS_DIR, "results.csv")
_MCNEMAR_CSV = os.path.join(_RESULTS_DIR, "mcnemar_results.csv")
_PLOTS_DIR   = os.path.join(_RESULTS_DIR, "plots")

MODEL_ORDER = ["LR", "NBOW", "DistilBERT"]
PTYPE_ORDER = ["punct_removal", "spelling_errors", "word_deletion", "word_order"]
PTYPE_LABELS = {
    "punct_removal": "Punct.\nRemoval",
    "spelling_errors": "Spelling\nErrors",
    "word_deletion": "Word\nDeletion",
    "word_order": "Word\nOrder",
}
SEVERITY_LEVELS = [0.0, 0.1, 0.3, 0.5]  # 0.0 = clean baseline

COMBINATION_ORDER = [
    "punct_plus_spelling",
    "deletion_plus_order",
    "spelling_plus_deletion",
    "all_combined",
]
COMBINATION_LABELS = {
    "punct_plus_spelling":    "Punct +\nSpelling",
    "deletion_plus_order":    "Deletion +\nOrder",
    "spelling_plus_deletion": "Spelling +\nDeletion",
    "all_combined":           "All\nCombined",
}

MODEL_COLORS = {
    "LR":          "#2196F3",   # blue
    "NBOW":        "#FF9800",   # orange
    "DistilBERT":  "#4CAF50",   # green
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results() -> pd.DataFrame:
    if not os.path.exists(_CSV):
        raise FileNotFoundError(
            f"Results file not found at {_CSV}.\n"
            "Run experiments/run_experiments.py first."
        )
    df = pd.read_csv(_CSV, dtype={"severity": str})
    df["severity_float"] = pd.to_numeric(df["severity"], errors="coerce")
    # Back-fill dataset column for CSVs written before multi-dataset support
    if "dataset" not in df.columns:
        df["dataset"] = "sst2"
    return df


def _save(fig: plt.Figure, name: str, dataset_id=None) -> None:
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    suffix = f"_{dataset_id}" if dataset_id else ""
    path = os.path.join(_PLOTS_DIR, f"{name}{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def _clean_acc(df: pd.DataFrame) -> dict[str, float]:
    """Return {model_name: clean_accuracy}."""
    clean = df[df["perturbation_type"] == "clean"]
    return dict(zip(clean["model"], clean["accuracy"]))


# ---------------------------------------------------------------------------
# Table 1 — Accuracy grid (Markdown)
# ---------------------------------------------------------------------------

def print_accuracy_table(df: pd.DataFrame, dataset_id: str = "") -> None:
    label = f" [{dataset_id.upper()}]" if dataset_id else ""
    print("\n" + "=" * 60)
    print(f"TABLE 1 — Validation Accuracy Grid{label}")
    print("=" * 60)

    # Build column labels
    col_labels = ["clean"]
    col_labels += ["punct_removal"]
    for ptype in ["spelling_errors", "word_deletion", "word_order"]:
        for sev in ["0.1", "0.3", "0.5"]:
            col_labels.append(f"{ptype}@{sev}")

    rows = []
    for model in MODEL_ORDER:
        mdf = df[df["model"] == model]
        row = [model]
        for label in col_labels:
            if label == "clean":
                val = mdf[mdf["perturbation_type"] == "clean"]["accuracy"].values
            elif label == "punct_removal":
                val = mdf[mdf["perturbation_type"] == "punct_removal"]["accuracy"].values
            else:
                ptype, sev = label.rsplit("@", 1)
                val = mdf[
                    (mdf["perturbation_type"] == ptype) & (mdf["severity"] == sev)
                ]["accuracy"].values
            row.append(f"{val[0]:.4f}" if len(val) > 0 else "—")
        rows.append(row)

    header = ["Model"] + col_labels
    col_w = [max(len(header[i]), max(len(r[i]) for r in rows)) + 2 for i in range(len(header))]

    def fmt_row(r):
        return "| " + " | ".join(r[i].ljust(col_w[i]) for i in range(len(r))) + " |"

    print(fmt_row(header))
    print("|" + "|".join("-" * (w + 2) for w in col_w) + "|")
    for row in rows:
        print(fmt_row(row))


# ---------------------------------------------------------------------------
# Plot 1 — Accuracy vs. severity line plots (2×2 grid)
# ---------------------------------------------------------------------------

def plot_accuracy_curves(df: pd.DataFrame, dataset_id: str = "") -> None:
    clean_acc = _clean_acc(df)

    ptypes_with_sev = ["spelling_errors", "word_deletion", "word_order"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()

    for ax_idx, ptype in enumerate(PTYPE_ORDER):
        ax = axes_flat[ax_idx]
        for model in MODEL_ORDER:
            mdf = df[df["model"] == model]
            xs = [0.0]  # clean baseline at severity=0
            ys = [clean_acc.get(model, np.nan)]

            if ptype == "punct_removal":
                # Single point — no severity axis
                row = mdf[mdf["perturbation_type"] == "punct_removal"]
                if not row.empty:
                    ys_single = row["accuracy"].values[0]
                    ax.axhline(
                        y=ys_single,
                        color=MODEL_COLORS[model],
                        linestyle="--",
                        linewidth=1.8,
                        label=f"{model} ({ys_single:.3f})",
                    )
                ax.axhline(
                    y=ys[0],
                    color=MODEL_COLORS[model],
                    linestyle=":",
                    linewidth=1,
                    alpha=0.5,
                )
            else:
                for sev in SEVERITIES:
                    row = mdf[
                        (mdf["perturbation_type"] == ptype) & (mdf["severity_float"] == sev)
                    ]
                    xs.append(sev)
                    ys.append(row["accuracy"].values[0] if not row.empty else np.nan)

                ax.plot(
                    xs, ys,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color=MODEL_COLORS[model],
                    label=model,
                )

        ax.set_title(PTYPE_LABELS[ptype].replace("\n", " "), fontsize=13, fontweight="bold")
        ax.set_xlabel("Severity" if ptype != "punct_removal" else "", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_ylim(0.4, 1.02)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        if ptype != "punct_removal":
            ax.set_xticks(SEVERITY_LEVELS)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Validation Accuracy vs. Perturbation Severity\nSST-2 Robustness Study"
        + (f" — {dataset_id.upper()}" if dataset_id else ""),
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, "accuracy_curves", dataset_id)


# ---------------------------------------------------------------------------
# Plot 2 — Heatmap of accuracy drop
# ---------------------------------------------------------------------------

def plot_drop_heatmap(df: pd.DataFrame, dataset_id: str = "") -> None:
    os.makedirs(_PLOTS_DIR, exist_ok=True)

    # Build a conditions list to use as x-axis
    conditions = []
    conditions.append(("punct_removal", "N/A"))
    for ptype in ["spelling_errors", "word_deletion", "word_order"]:
        for sev in ["0.1", "0.3", "0.5"]:
            conditions.append((ptype, sev))

    cond_labels = ["Punct\nRemoval"] \
        + [f"{p.replace('_', ' ').title()}\n@{s}" for p, s in conditions[1:]]

    matrix = np.full((len(MODEL_ORDER), len(conditions)), np.nan)
    for r_idx, model in enumerate(MODEL_ORDER):
        mdf = df[df["model"] == model]
        for c_idx, (ptype, sev) in enumerate(conditions):
            row = mdf[
                (mdf["perturbation_type"] == ptype) & (mdf["severity"] == sev)
            ]
            if not row.empty:
                matrix[r_idx, c_idx] = row["accuracy_drop"].values[0]

    # Clamp removed: negative drops (perturbation improves accuracy) are intentionally shown
    abs_max = max(0.05, np.nanmax(np.abs(matrix)))

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        xticklabels=cond_labels,
        yticklabels=MODEL_ORDER,
        center=0.0,
        vmin=-abs_max,
        vmax=abs_max,
        cbar_kws={"label": "Accuracy Drop (negative = improvement)"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(
        "Accuracy Drop from Clean Baseline (higher = more degraded)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Perturbation Condition", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    plt.tight_layout()
    _save(fig, "drop_heatmap", dataset_id)


# ---------------------------------------------------------------------------
# Plot 3 — Combination perturbations heatmap
# ---------------------------------------------------------------------------

def plot_combination_heatmap(df: pd.DataFrame, dataset_id: str = "") -> None:
    os.makedirs(_PLOTS_DIR, exist_ok=True)

    conditions = []
    for ctype in COMBINATION_ORDER:
        for sev in ["0.1", "0.3", "0.5"]:
            conditions.append((ctype, sev))

    # Check whether any combination rows exist
    combo_df = df[df["perturbation_type"].isin(COMBINATION_ORDER)]
    if combo_df.empty:
        print("[Plot 3] No combination perturbation rows in results — skipping.")
        return

    cond_labels = [
        f"{COMBINATION_LABELS[c].replace(chr(10), ' ')}\n@{s}"
        for c, s in conditions
    ]

    matrix = np.full((len(MODEL_ORDER), len(conditions)), np.nan)
    for r_idx, model in enumerate(MODEL_ORDER):
        mdf = df[df["model"] == model]
        for c_idx, (ctype, sev) in enumerate(conditions):
            row = mdf[
                (mdf["perturbation_type"] == ctype) & (mdf["severity"] == sev)
            ]
            if not row.empty:
                matrix[r_idx, c_idx] = row["accuracy_drop"].values[0]

    abs_max = max(0.05, np.nanmax(np.abs(matrix[~np.isnan(matrix)])) if not np.all(np.isnan(matrix)) else 0.05)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        xticklabels=cond_labels,
        yticklabels=MODEL_ORDER,
        center=0.0,
        vmin=-abs_max,
        vmax=abs_max,
        cbar_kws={"label": "Accuracy Drop (negative = improvement)"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(
        "Accuracy Drop — Combination Perturbations (higher = more degraded)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Perturbation Condition", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    plt.tight_layout()
    _save(fig, "combination_heatmap", dataset_id)


# ---------------------------------------------------------------------------
# Plot 4 — Per-class accuracy (positive vs. negative)
# ---------------------------------------------------------------------------

def plot_per_class_accuracy(df: pd.DataFrame, dataset_id: str = "") -> None:
    if "acc_neg" not in df.columns or "acc_pos" not in df.columns:
        print("[Plot 4] acc_neg/acc_pos columns not found — skipping.")
        return

    clean_neg = {}
    clean_pos = {}
    for model in MODEL_ORDER:
        clean_row = df[(df["model"] == model) & (df["perturbation_type"] == "clean")]
        if not clean_row.empty:
            clean_neg[model] = clean_row["acc_neg"].values[0]
            clean_pos[model] = clean_row["acc_pos"].values[0]

    ptypes = [p for p in PTYPE_ORDER if p != "punct_removal"]
    fig, axes = plt.subplots(1, len(ptypes), figsize=(14, 5), sharey=True)

    for ax, ptype in zip(axes, ptypes):
        for model in MODEL_ORDER:
            mdf = df[df["model"] == model]
            xs = [0.0]
            neg_ys = [clean_neg.get(model, np.nan)]
            pos_ys = [clean_pos.get(model, np.nan)]
            for sev in SEVERITIES:
                row = mdf[(mdf["perturbation_type"] == ptype) & (mdf["severity_float"] == sev)]
                xs.append(sev)
                neg_ys.append(row["acc_neg"].values[0] if not row.empty else np.nan)
                pos_ys.append(row["acc_pos"].values[0] if not row.empty else np.nan)
            color = MODEL_COLORS[model]
            ax.plot(xs, pos_ys, marker="o", markersize=4, linewidth=2,
                    color=color, label=f"{model} pos", linestyle="-")
            ax.plot(xs, neg_ys, marker="s", markersize=4, linewidth=2,
                    color=color, label=f"{model} neg", linestyle="--", alpha=0.7)
        ax.set_title(PTYPE_LABELS[ptype].replace("\n", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Severity", fontsize=9)
        ax.set_xticks(SEVERITY_LEVELS)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Per-Class Accuracy", fontsize=10)
    handles, labels_ = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=len(MODEL_ORDER) * 2,
               fontsize=8, bbox_to_anchor=(0.5, -0.08))
    ds_label = f" — {dataset_id.upper()}" if dataset_id else ""
    fig.suptitle(f"Per-Class Accuracy (solid=positive, dashed=negative){ds_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "per_class_accuracy", dataset_id)


# ---------------------------------------------------------------------------
# Plot 5 — Flip-rate vs. severity
# ---------------------------------------------------------------------------

def plot_flip_rate(df: pd.DataFrame, dataset_id: str = "") -> None:
    if "flip_rate" not in df.columns:
        print("[Plot 5] flip_rate column not found — skipping.")
        return

    ptypes = [p for p in PTYPE_ORDER if p != "punct_removal"]
    fig, axes = plt.subplots(1, len(ptypes), figsize=(14, 5), sharey=True)

    for ax, ptype in zip(axes, ptypes):
        for model in MODEL_ORDER:
            mdf = df[df["model"] == model]
            xs, ys = [0.0], [0.0]   # no flips at clean baseline
            for sev in SEVERITIES:
                row = mdf[(mdf["perturbation_type"] == ptype) & (mdf["severity_float"] == sev)]
                xs.append(sev)
                ys.append(row["flip_rate"].values[0] if not row.empty else np.nan)
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=2,
                    color=MODEL_COLORS[model], label=model)
        ax.set_title(PTYPE_LABELS[ptype].replace("\n", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Severity", fontsize=9)
        ax.set_xticks(SEVERITY_LEVELS)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0.0, 0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Flip Rate", fontsize=10)
    ds_label = f" — {dataset_id.upper()}" if dataset_id else ""
    fig.suptitle(f"Prediction Flip Rate vs. Perturbation Severity{ds_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "flip_rate", dataset_id)


# ---------------------------------------------------------------------------
# Plot 6 — DistilBERT confidence curves (cross-dataset)
# ---------------------------------------------------------------------------

def plot_bert_confidence(df: pd.DataFrame) -> None:
    if "mean_confidence" not in df.columns:
        print("[Plot 6] mean_confidence column not found — skipping.")
        return

    bdf = df[(df["model"] == "DistilBERT") & (df["mean_confidence"] != "")].copy()
    if bdf.empty:
        print("[Plot 6] No DistilBERT confidence rows — skipping.")
        return
    bdf["mean_confidence"] = pd.to_numeric(bdf["mean_confidence"], errors="coerce")
    bdf["std_confidence"]  = pd.to_numeric(bdf["std_confidence"],  errors="coerce")

    ptypes = [p for p in PTYPE_ORDER if p != "punct_removal"]
    datasets = sorted(bdf["dataset"].unique())
    fig, axes = plt.subplots(1, len(ptypes), figsize=(14, 5), sharey=True)

    dataset_colors = ["#1976D2", "#E65100", "#2E7D32"]
    for ax, ptype in zip(axes, ptypes):
        for d_idx, ds in enumerate(datasets):
            sub = bdf[(bdf["dataset"] == ds) & (bdf["perturbation_type"] == ptype)]
            clean_row = bdf[(bdf["dataset"] == ds) & (bdf["perturbation_type"] == "clean")]
            xs = [0.0]
            ys = [clean_row["mean_confidence"].values[0] if not clean_row.empty else np.nan]
            errs = [clean_row["std_confidence"].values[0]  if not clean_row.empty else 0.0]
            for sev in SEVERITIES:
                row = sub[sub["severity_float"] == sev]
                xs.append(sev)
                ys.append(row["mean_confidence"].values[0] if not row.empty else np.nan)
                errs.append(row["std_confidence"].values[0] if not row.empty else 0.0)
            color = dataset_colors[d_idx % len(dataset_colors)]
            ys_arr   = np.array(ys, dtype=float)
            errs_arr = np.array(errs, dtype=float)
            ax.plot(xs, ys_arr, marker="o", markersize=4, linewidth=2, color=color, label=ds)
            ax.fill_between(xs, ys_arr - errs_arr, ys_arr + errs_arr, color=color, alpha=0.15)
        ax.set_title(PTYPE_LABELS[ptype].replace("\n", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Severity", fontsize=9)
        ax.set_xticks(SEVERITY_LEVELS)
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Confidence (predicted label)", fontsize=10)
    fig.suptitle("DistilBERT Confidence Under Perturbation (mean ± std)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "bert_confidence")


# ---------------------------------------------------------------------------
# Plot 7 — Dataset comparison grouped bar
# ---------------------------------------------------------------------------

def plot_dataset_comparison(df: pd.DataFrame) -> None:
    datasets = sorted(df["dataset"].unique())
    if len(datasets) < 2:
        print("[Plot 7] Only one dataset in results — skipping dataset comparison.")
        return

    records = []
    for ds in datasets:
        sdf = df[df["dataset"] == ds]
        for model in MODEL_ORDER:
            mdf = sdf[sdf["model"] == model]
            if mdf.empty:
                continue
            clean = mdf[mdf["perturbation_type"] == "clean"]["accuracy"]
            mean_drop = mdf[mdf["perturbation_type"] != "clean"]["accuracy_drop"].mean()
            records.append({
                "dataset": ds, "model": model,
                "clean_acc": clean.values[0] if len(clean) else np.nan,
                "mean_drop": mean_drop,
            })

    rdf = pd.DataFrame(records)
    x = np.arange(len(MODEL_ORDER))
    width = 0.8 / len(datasets)
    offsets = np.linspace(-(len(datasets) - 1) / 2 * width, (len(datasets) - 1) / 2 * width, len(datasets))
    ds_colors = ["#1976D2", "#E65100", "#2E7D32"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for d_idx, (ds, color) in enumerate(zip(datasets, ds_colors)):
        ddf = rdf[rdf["dataset"] == ds]
        accs  = [ddf[ddf["model"] == m]["clean_acc"].values[0]  if not ddf[ddf["model"] == m].empty else np.nan for m in MODEL_ORDER]
        drops = [ddf[ddf["model"] == m]["mean_drop"].values[0]  if not ddf[ddf["model"] == m].empty else np.nan for m in MODEL_ORDER]
        ax1.bar(x + offsets[d_idx], accs,  width, label=ds, color=color, alpha=0.85)
        ax2.bar(x + offsets[d_idx], drops, width, label=ds, color=color, alpha=0.85)

    for ax, title, ylabel in [
        (ax1, "Clean Accuracy by Dataset", "Accuracy"),
        (ax2, "Mean Accuracy Drop Across All Conditions", "Mean Accuracy Drop"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cross-Dataset Comparison (trained on SST-2, evaluated OOD)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "dataset_comparison")


# ---------------------------------------------------------------------------
# Plot 8 — McNemar significance matrix
# ---------------------------------------------------------------------------

def plot_significance_matrix(dataset_id: str = "") -> None:
    if not os.path.exists(_MCNEMAR_CSV):
        print("[Plot 8] mcnemar_results.csv not found — skipping.")
        return

    mc = pd.read_csv(_MCNEMAR_CSV, dtype={"severity": str})
    if "dataset" in mc.columns and dataset_id:
        mc = mc[mc["dataset"] == dataset_id]
    if mc.empty:
        return

    model_pairs = sorted(mc[["model_a", "model_b"]].apply(
        lambda r: " vs ".join(sorted([r["model_a"], r["model_b"]])), axis=1
    ).unique())
    mc["pair"] = mc[["model_a", "model_b"]].apply(
        lambda r: " vs ".join(sorted([r["model_a"], r["model_b"]])), axis=1
    )
    mc["cond"] = mc["perturbation_type"] + "@" + mc["severity"]

    conditions_ordered = sorted(mc["cond"].unique())
    matrix = np.full((len(model_pairs), len(conditions_ordered)), np.nan)
    sig_mask = np.zeros_like(matrix, dtype=bool)

    for r_idx, pair in enumerate(model_pairs):
        for c_idx, cond in enumerate(conditions_ordered):
            row = mc[(mc["pair"] == pair) & (mc["cond"] == cond)]
            if not row.empty:
                matrix[r_idx, c_idx] = row["p_value"].values[0]
                sig_mask[r_idx, c_idx] = bool(row["significant"].values[0])

    fig, ax = plt.subplots(figsize=(max(14, len(conditions_ordered) * 0.6), max(4, len(model_pairs) * 1.2)))
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=0.0, vmax=0.1, center=0.05,
        xticklabels=conditions_ordered, yticklabels=model_pairs,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "p-value"},
    )
    # Overlay * for significant cells
    for r_idx in range(len(model_pairs)):
        for c_idx in range(len(conditions_ordered)):
            if sig_mask[r_idx, c_idx]:
                ax.text(c_idx + 0.5, r_idx + 0.15, "*", ha="center", va="top",
                        fontsize=14, fontweight="bold", color="black")
    ds_label = f" — {dataset_id.upper()}" if dataset_id else ""
    ax.set_title(f"McNemar\u2019s Test p-values (\u2605 = p < 0.05){ds_label}",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Perturbation Condition", fontsize=10)
    ax.set_ylabel("Model Pair", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    _save(fig, "significance_matrix", dataset_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = _load_results()
    datasets = sorted(df["dataset"].unique())

    for dataset_id in datasets:
        sub = df[df["dataset"] == dataset_id]
        print(f"\n{'='*60}\nDATASET: {dataset_id.upper()}\n{'='*60}")
        print_accuracy_table(sub, dataset_id)
        plot_accuracy_curves(sub, dataset_id)
        plot_drop_heatmap(sub, dataset_id)
        plot_combination_heatmap(sub, dataset_id)
        plot_per_class_accuracy(sub, dataset_id)
        plot_flip_rate(sub, dataset_id)
        plot_significance_matrix(dataset_id)

    # Cross-dataset plots (run once over full df)
    plot_bert_confidence(df)
    plot_dataset_comparison(df)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
