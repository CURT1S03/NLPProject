"""
analyze_results.py
Loads results/results.csv and produces:

  Table 1  — markdown accuracy grid (models × conditions)
  Plot 1   — 2×2 line plots: accuracy vs. severity per perturbation type
  Plot 2   — heatmap of accuracy drop (models × perturbation conditions)

All figures are saved to results/plots/.

Usage
-----
  python -m experiments.analyze_results
  python experiments/analyze_results.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
_CSV = os.path.join(_RESULTS_DIR, "results.csv")
_PLOTS_DIR = os.path.join(_RESULTS_DIR, "plots")

MODEL_ORDER = ["LR", "NBOW", "DistilBERT"]
PTYPE_ORDER = ["punct_removal", "spelling_errors", "word_deletion", "word_order"]
PTYPE_LABELS = {
    "punct_removal": "Punct.\nRemoval",
    "spelling_errors": "Spelling\nErrors",
    "word_deletion": "Word\nDeletion",
    "word_order": "Word\nOrder",
}
SEVERITY_LEVELS = [0.0, 0.1, 0.3, 0.5]  # 0.0 = clean baseline

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
    df = pd.read_csv(_CSV)
    # Coerce severity to float where possible, keep "N/A" as NaN
    df["severity_float"] = pd.to_numeric(df["severity"], errors="coerce")
    return df


def _clean_acc(df: pd.DataFrame) -> dict[str, float]:
    """Return {model_name: clean_accuracy}."""
    clean = df[df["perturbation_type"] == "clean"]
    return dict(zip(clean["model"], clean["accuracy"]))


# ---------------------------------------------------------------------------
# Table 1 — Accuracy grid (Markdown)
# ---------------------------------------------------------------------------

def print_accuracy_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("TABLE 1 — Validation Accuracy Grid")
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

def plot_accuracy_curves(df: pd.DataFrame) -> None:
    os.makedirs(_PLOTS_DIR, exist_ok=True)
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
        "Validation Accuracy vs. Perturbation Severity\nSTT-2 Robustness Study",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(_PLOTS_DIR, "accuracy_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot 1] Saved to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2 — Heatmap of accuracy drop
# ---------------------------------------------------------------------------

def plot_drop_heatmap(df: pd.DataFrame) -> None:
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

    # Clamp negatives (tiny rounding artefacts) to zero
    matrix = np.clip(matrix, 0, None)

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=cond_labels,
        yticklabels=MODEL_ORDER,
        vmin=0.0,
        vmax=max(0.3, np.nanmax(matrix)),
        cbar_kws={"label": "Accuracy Drop"},
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
    path = os.path.join(_PLOTS_DIR, "drop_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot 2] Saved to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = _load_results()
    print_accuracy_table(df)
    plot_accuracy_curves(df)
    plot_drop_heatmap(df)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
