# NLPProject — SST-2 Robustness Study

Evaluates how three NLP models (Logistic Regression, Neural Bag-of-Words, DistilBERT) handle synthetic text perturbations on the [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) sentiment classification dataset.

---

## Repo Structure

```
NLPProject/
├── data/
│   ├── load_data.py          # HuggingFace SST-2 loader with module-level cache
│   └── perturbations.py      # All perturbation types + perturb() public API
├── models/
│   ├── baseline_lr.py        # TF-IDF (1-2 ngrams, 50k feat) + Logistic Regression
│   ├── baseline_nbow.py      # GloVe-100d averaged vectors + 2-layer MLP
│   └── bert_eval.py          # distilbert-base-uncased-finetuned-sst-2-english (zero-shot)
├── experiments/
│   ├── run_experiments.py    # Sweeps all 23 model × condition combinations → results.csv
│   └── analyze_results.py    # Accuracy table + 3 plots
├── results/
│   ├── perturbations/
│   │   └── validation_perturbations.csv   # Pre-generated perturbed sentences (872 × 26 conditions)
│   ├── plots/                             # Created on first analysis run
│   │   ├── accuracy_curves.png
│   │   ├── drop_heatmap.png
│   │   └── combination_heatmap.png
│   ├── results.csv                        # Experiment output (model × condition accuracies)
│   ├── lr_model.joblib                    # Cached LR pipeline (auto-generated)
│   └── nbow_model.pt                      # Cached NBOW checkpoint (auto-generated)
├── NLPProject.ipynb           # End-to-end notebook (Colab-compatible)
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
# Download and extract into data/
curl -O https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt -d data/
```

Or set the `GLOVE_PATH` environment variable to an existing `glove.6B.100d.txt` path.

---

## Running the Pipeline

Run each step from the **project root** in order:

```bash
# Step 1 — Generate all perturbed validation sentences (saves to results/perturbations/)
python -m data.perturbations

# Step 2 — Train LR + NBOW baselines (DistilBERT needs no training)
python -m models.baseline_lr
python -m models.baseline_nbow

# Step 3 — Sweep all 23 conditions across all 3 models
python -m experiments.run_experiments

# Step 4 — Produce accuracy table + 3 plots
python -m experiments.analyze_results
```

Skip slow models during development:
```bash
python -m experiments.run_experiments --skip-nbow --skip-bert
```

---

## Perturbation Types

| `ptype` | Severity axis | Description |
|---|---|---|
| `punct_removal` | Binary (always full) | Strips all `string.punctuation` characters |
| `spelling_errors` | Fraction of words corrupted | Per-word: random character swap / deletion / insertion / adjacent-transposition |
| `word_deletion` | Fraction of tokens dropped | Removes grammatical function words only (articles, prepositions, auxiliaries); negation words protected |
| `word_order` | Fraction of adjacent pairs swapped | Iterates tokens, swaps adjacent pair with probability = severity |
| `punct_plus_spelling` | Combined | Punctuation removal → spelling errors |
| `deletion_plus_order` | Combined | Word deletion → word-order disruption |
| `spelling_plus_deletion` | Combined | Spelling errors → word deletion |
| `all_combined` | Combined | All four steps applied in sequence |

Severities used for types 2–4 and all combinations: **0.1, 0.3, 0.5**.  
`punct_removal` is always full (severity = N/A).

Total conditions per model: **23** (1 clean + 1 punct_removal + 9 single-severity + 12 combination-severity).

---

## Results

`results/results.csv` columns:

| Column | Description |
|---|---|
| `model` | `LR`, `NBOW`, or `DistilBERT` |
| `perturbation_type` | One of the 8 ptype names above, or `clean` |
| `severity` | `0.1`, `0.3`, `0.5`, or `N/A` |
| `accuracy` | Fraction of 872 val examples correctly labelled |
| `accuracy_drop` | `clean_accuracy − perturbed_accuracy` (negative = perturbation helped) |
| `f1` | Binary F1 score |
| `f1_drop` | `clean_f1 − perturbed_f1` |

### Plots

| File | Contents |
|---|---|
| `accuracy_curves.png` | 2×2 grid — accuracy vs. severity for the 4 atomic perturbation types, one line per model |
| `drop_heatmap.png` | Heatmap — accuracy drop for atomic conditions; diverging colormap (green = improvement, red = degradation) |
| `combination_heatmap.png` | Same diverging heatmap for the 4 combination conditions |

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
