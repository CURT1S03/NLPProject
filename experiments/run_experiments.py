"""
run_experiments.py
Sweeps all model × perturbation-condition combinations and records accuracy.

Evaluation conditions
---------------------
  1 clean baseline  +  4 perturbation types  ×  3 severity levels  =  13 conditions per model
  3 models  →  39 rows in results/results.csv

Output
------
  results/results.csv   — one row per (model, condition)
  Columns: model, perturbation_type, severity, accuracy, accuracy_drop

Usage
-----
  Run from the project root:
      python -m experiments.run_experiments
  or simply:
      python experiments/run_experiments.py
"""

import os
import sys
import csv
import time

# Allow running from project root without installing as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.load_data import get_val_sentences
from data.perturbations import perturb, PERTURBATION_TYPES, SEVERITIES
from models.baseline_lr import LogisticRegressionModel
from models.baseline_nbow import NBOWModel
from models.bert_eval import DistilBERTModel
from sklearn.metrics import accuracy_score

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
_RESULTS_CSV = os.path.join(_RESULTS_DIR, "results.csv")

# ---------------------------------------------------------------------------
# Evaluation conditions
# ---------------------------------------------------------------------------
# Each condition is (perturbation_type, severity).
# "clean" is represented as (None, None).
# punct_removal is binary — severity is N/A, represented as "N/A".

def _build_conditions() -> list[tuple]:
    """Returns list of (ptype_label, ptype_arg, severity_label, severity_arg)."""
    conditions = [("clean", None, "N/A", None)]
    for ptype in PERTURBATION_TYPES:
        if ptype == "punct_removal":
            conditions.append(("punct_removal", "punct_removal", "N/A", None))
        else:
            for sev in SEVERITIES:
                conditions.append((ptype, ptype, str(sev), sev))
    return conditions  # 1 + 1 + 3 + 3 + 3 = 13 conditions


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _evaluate_model(
    model_name: str,
    predict_fn,
    val_sents: list[str],
    val_labels: list[int],
    conditions: list[tuple],
    clean_acc: float,
) -> list[dict]:
    rows = []
    for ptype_label, ptype_arg, sev_label, sev_arg in conditions:
        if ptype_arg is None:
            sentences = val_sents
        else:
            sentences = perturb(val_sents, ptype=ptype_arg, severity=sev_arg)

        preds = predict_fn(sentences)
        acc = accuracy_score(val_labels, preds)
        drop = clean_acc - acc

        rows.append({
            "model": model_name,
            "perturbation_type": ptype_label,
            "severity": sev_label,
            "accuracy": round(acc, 4),
            "accuracy_drop": round(drop, 4),
        })

        flag = f"  [{model_name}] {ptype_label} (sev={sev_label})  acc={acc:.4f}  drop={drop:+.4f}"
        print(flag)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(skip_nbow: bool = False, skip_bert: bool = False) -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    conditions = _build_conditions()

    val_sents, val_labels = get_val_sentences()
    # Convert to plain list in case HuggingFace returns a custom sequence type
    val_sents = list(val_sents)
    val_labels = list(val_labels)

    all_rows: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Logistic Regression
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 1: Logistic Regression + TF-IDF")
    print("=" * 60)
    lr_model = LogisticRegressionModel()
    lr_model.train()
    # Measure clean accuracy first (needed for drop calculation)
    lr_clean_preds = lr_model.predict(val_sents)
    lr_clean_acc = accuracy_score(val_labels, lr_clean_preds)
    print(f"  [LR] Clean accuracy: {lr_clean_acc:.4f}")

    lr_rows = _evaluate_model(
        "LR", lr_model.predict, val_sents, val_labels, conditions, lr_clean_acc
    )
    all_rows.extend(lr_rows)

    # ------------------------------------------------------------------
    # 2. NBOW
    # ------------------------------------------------------------------
    if not skip_nbow:
        print("\n" + "=" * 60)
        print("MODEL 2: NBOW (GloVe-100d + MLP)")
        print("=" * 60)
        nbow_model = NBOWModel()
        nbow_model.train()
        nbow_clean_preds = nbow_model.predict(val_sents)
        nbow_clean_acc = accuracy_score(val_labels, nbow_clean_preds)
        print(f"  [NBOW] Clean accuracy: {nbow_clean_acc:.4f}")

        nbow_rows = _evaluate_model(
            "NBOW", nbow_model.predict, val_sents, val_labels, conditions, nbow_clean_acc
        )
        all_rows.extend(nbow_rows)
    else:
        print("\n[Skipping NBOW]")

    # ------------------------------------------------------------------
    # 3. DistilBERT
    # ------------------------------------------------------------------
    if not skip_bert:
        print("\n" + "=" * 60)
        print("MODEL 3: DistilBERT (pre-fine-tuned on SST-2)")
        print("=" * 60)
        bert_model = DistilBERTModel()
        bert_model.load()
        bert_clean_preds = bert_model.predict(val_sents)
        bert_clean_acc = accuracy_score(val_labels, bert_clean_preds)
        print(f"  [BERT] Clean accuracy: {bert_clean_acc:.4f}")

        bert_rows = _evaluate_model(
            "DistilBERT", bert_model.predict, val_sents, val_labels, conditions, bert_clean_acc
        )
        all_rows.extend(bert_rows)
    else:
        print("\n[Skipping DistilBERT]")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = ["model", "perturbation_type", "severity", "accuracy", "accuracy_drop"]
    with open(_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {_RESULTS_CSV}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NLP robustness experiments.")
    parser.add_argument("--skip-nbow", action="store_true", help="Skip NBOW model evaluation.")
    parser.add_argument("--skip-bert", action="store_true", help="Skip DistilBERT evaluation.")
    args = parser.parse_args()

    t0 = time.time()
    run(skip_nbow=args.skip_nbow, skip_bert=args.skip_bert)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
