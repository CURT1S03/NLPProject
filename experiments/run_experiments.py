"""
run_experiments.py
Sweeps all model × dataset × perturbation-condition combinations and records metrics.

Models are trained once on SST-2 (Option A: cross-dataset OOD study), then
evaluated against all three datasets.

Evaluation conditions: 23 per model per dataset.
Output files:
  results/results.csv         — one row per (dataset, model, condition)
  results/mcnemar_results.csv — McNemar test between model pairs per condition

Usage:
  python -m experiments.run_experiments
  python -m experiments.run_experiments --skip-nbow --skip-bert
  python -m experiments.run_experiments --datasets sst2 imdb
"""

import os
import sys
import csv
import time
import random
import numpy as np

# Allow running from project root without installing as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.load_data import get_val_sentences, list_datasets
from data.perturbations import perturb, PERTURBATION_TYPES, SEVERITIES, SEED, normalize_sentence
from models.baseline_lr import LogisticRegressionModel
from models.baseline_nbow import NBOWModel
from models.bert_eval import DistilBERTModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
_RESULTS_CSV = os.path.join(_RESULTS_DIR, "results.csv")
_MCNEMAR_CSV = os.path.join(_RESULTS_DIR, "mcnemar_results.csv")

COMBINATION_TYPES = [
    "punct_plus_spelling",
    "deletion_plus_order",
    "spelling_plus_deletion",
    "all_combined",
]

DATASETS = list_datasets()   # ["sst2", "imdb", "yelp_polarity"]
MAX_VAL_EXAMPLES = 872       # cap IMDb/Yelp to match SST-2 val size

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
    for ctype in COMBINATION_TYPES:
        for sev in SEVERITIES:
            conditions.append((ctype, ctype, str(sev), sev))
    return conditions  # 1 + 1 + 9 + 12 = 23 conditions


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
    clean_f1: float,
    clean_preds: np.ndarray,
    dataset: str,
    score_fn=None,           # optional callable returning (preds, scores)
) -> tuple[list[dict], dict]:
    rows = []
    preds_by_condition: dict[tuple, np.ndarray] = {}

    for ptype_label, ptype_arg, sev_label, sev_arg in conditions:
        if ptype_arg is None:
            sentences = val_sents
        else:
            random.seed(SEED)
            sentences = perturb(val_sents, ptype=ptype_arg, severity=sev_arg)

        if score_fn is not None:
            preds, scores = score_fn(sentences)
            mean_conf = float(np.mean(scores))
            std_conf  = float(np.std(scores))
        else:
            preds = predict_fn(sentences)
            mean_conf = float("nan")
            std_conf  = float("nan")

        preds_by_condition[(ptype_label, sev_label)] = preds

        acc    = accuracy_score(val_labels, preds)
        f1     = f1_score(val_labels, preds)
        drop   = clean_acc - acc
        f1_drop = clean_f1 - f1
        flip   = 0.0 if ptype_arg is None else float(np.mean(preds != clean_preds))

        tn, fp, fn, tp = confusion_matrix(val_labels, preds, labels=[0, 1]).ravel()
        acc_neg = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        acc_pos = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        rows.append({
            "dataset": dataset,
            "model": model_name,
            "perturbation_type": ptype_label,
            "severity": sev_label,
            "accuracy": round(acc, 4),
            "accuracy_drop": round(drop, 4),
            "f1": round(f1, 4),
            "f1_drop": round(f1_drop, 4),
            "acc_neg": round(acc_neg, 4),
            "acc_pos": round(acc_pos, 4),
            "flip_rate": round(flip, 4),
            "mean_confidence": round(mean_conf, 4) if not (mean_conf != mean_conf) else "",
            "std_confidence":  round(std_conf, 4)  if not (std_conf  != std_conf)  else "",
        })

        print(
            f"  [{model_name}|{dataset}] {ptype_label} (sev={sev_label})"
            f"  acc={acc:.4f}  drop={drop:+.4f}  f1={f1:.4f}"
            f"  flip={flip:.3f}  neg={acc_neg:.3f}  pos={acc_pos:.3f}"
        )

    return rows, preds_by_condition


# ---------------------------------------------------------------------------
# McNemar significance test
# ---------------------------------------------------------------------------

def _compute_mcnemar(
    preds_by_model: dict[str, dict[tuple, np.ndarray]],
    val_labels: list[int],
    dataset: str,
) -> list[dict]:
    """For every (model_a, model_b, condition) triple, run McNemar's test."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
    except ImportError:
        print("[McNemar] statsmodels not installed — skipping. Run: pip install statsmodels")
        return []

    from itertools import combinations
    rows = []
    models = list(preds_by_model.keys())
    conditions = list(next(iter(preds_by_model.values())).keys())
    labels_arr = np.array(val_labels)

    for condition in conditions:
        for m_a, m_b in combinations(models, 2):
            preds_a = preds_by_model[m_a][condition]
            preds_b = preds_by_model[m_b][condition]
            b = int(np.sum((preds_a == labels_arr) & (preds_b != labels_arr)))
            c = int(np.sum((preds_a != labels_arr) & (preds_b == labels_arr)))
            if b + c == 0:
                continue   # no disagreements — test undefined
            result = _mcnemar([[0, b], [c, 0]], exact=False, correction=True)
            rows.append({
                "dataset":          dataset,
                "perturbation_type": condition[0],
                "severity":          condition[1],
                "model_a":           m_a,
                "model_b":           m_b,
                "statistic":         round(result.statistic, 4),
                "p_value":           round(result.pvalue, 4),
                "significant":       result.pvalue < 0.05,
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    skip_nbow: bool = False,
    skip_bert: bool = False,
    datasets: list[str] | None = None,
) -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    conditions = _build_conditions()
    target_datasets = datasets if datasets else DATASETS

    all_rows: list[dict] = []
    all_mcnemar_rows: list[dict] = []

    # ------------------------------------------------------------------
    # Train models once on SST-2 (cross-dataset OOD evaluation)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING MODELS ON SST-2")
    print("=" * 60)

    lr_model = LogisticRegressionModel()
    lr_model.train()

    nbow_model = NBOWModel() if not skip_nbow else None
    if nbow_model:
        nbow_model.train()

    bert_model = DistilBERTModel() if not skip_bert else None
    if bert_model:
        bert_model.load()

    # ------------------------------------------------------------------
    # Evaluate on each dataset
    # ------------------------------------------------------------------
    for dataset_id in target_datasets:
        print("\n" + "=" * 60)
        print(f"DATASET: {dataset_id.upper()}")
        print("=" * 60)

        val_sents, val_labels = get_val_sentences(
            dataset_id=dataset_id,
            max_examples=MAX_VAL_EXAMPLES,
        )
        val_sents  = [normalize_sentence(s) for s in val_sents]
        val_labels = list(val_labels)
        dataset_preds_by_model: dict[str, dict] = {}

        # LR
        print(f"\n  [LR | {dataset_id}]")
        lr_clean_preds = lr_model.predict(val_sents)
        lr_clean_acc   = accuracy_score(val_labels, lr_clean_preds)
        lr_clean_f1    = f1_score(val_labels, lr_clean_preds)
        print(f"    Clean acc={lr_clean_acc:.4f}  f1={lr_clean_f1:.4f}")
        lr_rows, lr_preds = _evaluate_model(
            "LR", lr_model.predict, val_sents, val_labels, conditions,
            lr_clean_acc, lr_clean_f1, lr_clean_preds, dataset_id,
        )
        all_rows.extend(lr_rows)
        dataset_preds_by_model["LR"] = lr_preds

        # NBOW
        if nbow_model:
            print(f"\n  [NBOW | {dataset_id}]")
            nbow_clean_preds = nbow_model.predict(val_sents)
            nbow_clean_acc   = accuracy_score(val_labels, nbow_clean_preds)
            nbow_clean_f1    = f1_score(val_labels, nbow_clean_preds)
            print(f"    Clean acc={nbow_clean_acc:.4f}  f1={nbow_clean_f1:.4f}")
            nbow_rows, nbow_preds = _evaluate_model(
                "NBOW", nbow_model.predict, val_sents, val_labels, conditions,
                nbow_clean_acc, nbow_clean_f1, nbow_clean_preds, dataset_id,
            )
            all_rows.extend(nbow_rows)
            dataset_preds_by_model["NBOW"] = nbow_preds
        else:
            print(f"\n  [Skipping NBOW]")

        # DistilBERT
        if bert_model:
            print(f"\n  [DistilBERT | {dataset_id}]")
            bert_clean_preds, bert_clean_scores = bert_model.predict_with_scores(val_sents)
            bert_clean_acc = accuracy_score(val_labels, bert_clean_preds)
            bert_clean_f1  = f1_score(val_labels, bert_clean_preds)
            print(f"    Clean acc={bert_clean_acc:.4f}  f1={bert_clean_f1:.4f}  conf={bert_clean_scores.mean():.4f}")
            bert_rows, bert_preds = _evaluate_model(
                "DistilBERT", bert_model.predict, val_sents, val_labels, conditions,
                bert_clean_acc, bert_clean_f1, bert_clean_preds, dataset_id,
                score_fn=bert_model.predict_with_scores,
            )
            all_rows.extend(bert_rows)
            dataset_preds_by_model["DistilBERT"] = bert_preds
        else:
            print(f"\n  [Skipping DistilBERT]")

        # McNemar
        if len(dataset_preds_by_model) >= 2:
            mc_rows = _compute_mcnemar(dataset_preds_by_model, val_labels, dataset_id)
            all_mcnemar_rows.extend(mc_rows)

    # ------------------------------------------------------------------
    # Write results.csv
    # ------------------------------------------------------------------
    fieldnames = [
        "dataset", "model", "perturbation_type", "severity",
        "accuracy", "accuracy_drop", "f1", "f1_drop",
        "acc_neg", "acc_pos", "flip_rate",
        "mean_confidence", "std_confidence",
    ]
    with open(_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to {_RESULTS_CSV}  ({len(all_rows)} rows)")

    # ------------------------------------------------------------------
    # Write mcnemar_results.csv
    # ------------------------------------------------------------------
    if all_mcnemar_rows:
        mc_fieldnames = [
            "dataset", "perturbation_type", "severity",
            "model_a", "model_b", "statistic", "p_value", "significant",
        ]
        with open(_MCNEMAR_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mc_fieldnames)
            writer.writeheader()
            writer.writerows(all_mcnemar_rows)
        print(f"McNemar results saved to {_MCNEMAR_CSV}  ({len(all_mcnemar_rows)} rows)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NLP robustness experiments.")
    parser.add_argument("--skip-nbow", action="store_true", help="Skip NBOW model evaluation.")
    parser.add_argument("--skip-bert", action="store_true", help="Skip DistilBERT evaluation.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to evaluate (default: all). E.g. --datasets sst2 imdb")
    args = parser.parse_args()

    t0 = time.time()
    run(skip_nbow=args.skip_nbow, skip_bert=args.skip_bert, datasets=args.datasets)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
