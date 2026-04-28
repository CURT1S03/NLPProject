# NLPProject — Sentiment Robustness Study

Evaluates how three NLP models (Logistic Regression, Neural Bag-of-Words, DistilBERT) handle synthetic text perturbations across three sentiment classification datasets: [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2), IMDB, and Yelp Polarity.

Models are trained once on SST-2 and evaluated on all three datasets (cross-dataset OOD study). Each model is tested across **23 conditions** per dataset: 1 clean baseline, 4 atomic perturbation types (3 severities each), and 4 combination perturbation types (3 severities each).

---

## Repo Structure

```
NLPProject/
├── data/
│   ├── __init__.py
│   ├── load_data.py          # Loads SST-2, IMDB, and Yelp Polarity via HuggingFace datasets library
│   └── perturbations.py      # 4 atomic + 4 combination perturbation types
├── models/
│   ├── __init__.py
│   ├── baseline_lr.py        # TF-IDF (1-2 ngrams, 50k feat) + Logistic Regression
│   ├── baseline_nbow.py      # GloVe-100d averaged vectors + 2-layer MLP
│   └── bert_eval.py          # distilbert-base-uncased-finetuned-sst-2-english
├── experiments/
│   ├── __init__.py
│   ├── run_experiments.py    # Sweeps all model × dataset × condition → results.csv + mcnemar_results.csv
│   └── analyze_results.py    # Accuracy tables + 7 plot types per dataset
├── results/
│   ├── perturbations/        # Pre-generated perturbed sentences
│   ├── plots/                # Per-dataset plots (created on first analysis run)
│   │   ├── accuracy_curves_{dataset}.png
│   │   ├── drop_heatmap_{dataset}.png
│   │   ├── combination_heatmap_{dataset}.png
│   │   ├── per_class_accuracy_{dataset}.png
│   │   ├── flip_rate_{dataset}.png
│   │   ├── bert_confidence.png
│   │   └── dataset_comparison.png
│   ├── results.csv           # All metrics: model × dataset × condition
│   ├── mcnemar_results.csv   # McNemar significance tests between model pairs (generated if statsmodels installed)
│   ├── lr_model.joblib       # Cached LR pipeline (auto-generated, gitignored)
│   └── nbow_model.pt         # Cached NBOW checkpoint (auto-generated, gitignored)
├── NLPProject.ipynb          # End-to-end notebook (Colab-compatible)
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download GloVe embeddings (required for NBOW only)

```bash
curl -L https://nlp.stanford.edu/data/glove.6B.zip -o data/glove.6B.zip
unzip data/glove.6B.zip glove.6B.100d.txt -d data/
rm data/glove.6B.zip
```

---

## Running the Pipeline

Run from the **project root**:

```bash
# Run full experiment across all 3 models × 3 datasets × 23 conditions
python -m experiments.run_experiments

# Generate all tables and plots
python -m experiments.analyze_results
```

**Flags:**
```bash
python -m experiments.run_experiments --skip-nbow          # skip NBOW (no GloVe needed)
python -m experiments.run_experiments --skip-bert          # skip DistilBERT (faster)
python -m experiments.run_experiments --datasets sst2      # run on one dataset only
```

---

## Perturbation Types

### Atomic (4 types)

| `ptype` | Severity | Description |
|---|---|---|
| `punct_removal` | N/A (always full) | Strips all punctuation |
| `spelling_errors` | 0.1 / 0.3 / 0.5 | Random character-level noise per word (swap, delete, insert, transpose) |
| `word_deletion` | 0.1 / 0.3 / 0.5 | Drops function words (articles, prepositions, auxiliaries); negation protected |
| `word_order` | 0.1 / 0.3 / 0.5 | Swaps adjacent token pairs with given probability |

### Combination (4 types, severity 0.1 / 0.3 / 0.5)

| `ptype` | Steps |
|---|---|
| `punct_plus_spelling` | punct_removal → spelling_errors |
| `deletion_plus_order` | word_deletion → word_order |
| `spelling_plus_deletion` | spelling_errors → word_deletion |
| `all_combined` | All four atomic steps in sequence |

**Total conditions per model per dataset: 23**

---

## Metrics

`results/results.csv` columns:

| Column | Description |
|---|---|
| `dataset` | `sst2`, `imdb`, or `yelp_polarity` |
| `model` | `LR`, `NBOW`, or `DistilBERT` |
| `perturbation_type` | Perturbation name or `clean` |
| `severity` | `0.1`, `0.3`, `0.5`, or `N/A` |
| `accuracy` | Fraction correctly labelled |
| `accuracy_drop` | `clean_accuracy − perturbed_accuracy` |
| `f1` | Binary F1 score |
| `f1_drop` | `clean_f1 − perturbed_f1` |
| `acc_neg` | Accuracy on negative class only |
| `acc_pos` | Accuracy on positive class only |
| `flip_rate` | Fraction of predictions that changed vs. clean |
| `mean_confidence` | Mean softmax confidence (DistilBERT only) |
| `std_confidence` | Std of softmax confidence (DistilBERT only) |

### Plots

| File | Contents |
|---|---|
| `accuracy_curves_{dataset}.png` | Accuracy vs. severity per perturbation type, one line per model |
| `drop_heatmap_{dataset}.png` | Accuracy drop heatmap for atomic conditions |
| `combination_heatmap_{dataset}.png` | Accuracy drop heatmap for combination conditions |
| `per_class_accuracy_{dataset}.png` | Positive vs. negative class accuracy per condition |
| `flip_rate_{dataset}.png` | Prediction flip rate per condition |
| `bert_confidence.png` | DistilBERT confidence distribution across conditions |
| `dataset_comparison.png` | Cross-dataset accuracy comparison |

---

## Datasets

| Dataset | Examples | Avg. Length | Labels | Source |
|---|---|---|---|---|
| SST-2 | 872 (val) | 19.5 tokens | Movie reviews (phrases) | Socher et al., EMNLP 2013 |
| IMDB | 25,000 (test) | ~230 tokens | Movie reviews (full) | Maas et al., ACL 2011 |
| Yelp Polarity | 38,000 (test) | ~155 tokens | Yelp business reviews | Zhang et al., NeurIPS 2015 |

All datasets use binary labels (0 = negative, 1 = positive). IMDB and Yelp are subsampled to 872 examples using a fixed random seed for comparability with SST-2.

### References

- Socher, R., et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP 2013.
- Maas, A., et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL 2011.
- Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015.

---

## Changelog

All changes made in the Phase 2 implementation pass (April 2026):

### Bug fixes (blockers)

| File | Change | Reason |
|---|---|---|
| `data/perturbations.py` | Added `PERTURBATION_TYPES` constant | `run_experiments.py` imported it but it was never defined → `ImportError` on every run |
| `experiments/run_experiments.py` | Added `import random`; added `random.seed(SEED)` before each `perturb()` call | Stochastic perturbations (spelling, deletion, word order) produced different results on every run — results were not reproducible |
| `experiments/run_experiments.py` | Added `COMBINATION_TYPES`; expanded `_build_conditions()` to include all 12 combination-severity conditions | 12 of the 26 generated conditions were never evaluated on any model |
| `experiments/run_experiments.py` | Imported `normalize_sentence`; replaced `val_sents = list(val_sents)` with normalization call | `save_validation_perturbations()` normalizes before perturbing, but live evaluation did not — models received differently preprocessed text than the CSV stored |

### Metric improvements

| File | Change | Reason |
|---|---|---|
| `experiments/run_experiments.py` | Added `f1_score` import; computed `f1` and `f1_drop` per condition; added both to CSV | Accuracy alone can mask class-imbalance effects; F1 gives a more complete picture of degradation |

### Analysis / plot fixes

| File | Change | Reason |
|---|---|---|
| `experiments/analyze_results.py` | Fixed plot title `"STT-2"` → `"SST-2"` | Typo |
| `experiments/analyze_results.py` | Removed `np.clip(matrix, 0, None)`; changed heatmap to diverging `RdYlGn_r` colormap centered at 0 | Clipping hid cases where a perturbation *improved* accuracy (e.g. `punct_removal` helping NBOW via better GloVe coverage) — these are valid research findings |
| `experiments/analyze_results.py` | Added `COMBINATION_ORDER`, `COMBINATION_LABELS`, `plot_combination_heatmap()`, and called it in `main()` | Combination conditions were generated and included in `results.csv` but never visualised |

### Code quality

| File | Change | Reason |
|---|---|---|
| `models/baseline_nbow.py` | Added `import string`; fixed `sentences_to_embeddings()` to strip punctuation from tokens before GloVe lookup | Whitespace-only split left punctuation attached to tokens (e.g. `"great."`) causing silent OOV misses in GloVe; this also meant `punct_removal` *artifactually helped* NBOW by improving token coverage |
| `models/baseline_nbow.py` | Added `weights_only=True` to `torch.load()` | Removes deprecation warning in PyTorch ≥ 2.0; prevents arbitrary code execution if the cache file is untrusted |
