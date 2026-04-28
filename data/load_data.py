"""
load_data.py
Loads sentiment datasets from HuggingFace.

Supported datasets: sst2, imdb, yelp_polarity
All return binary labels (0 = negative, 1 = positive).

Note: SST-2 test labels are hidden (-1), so evaluation uses the validation
split (872 examples). IMDb and Yelp Polarity use their test splits;
max_val_examples caps the size for comparability.
"""

import random as _random
from datasets import load_dataset
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_DATASET_REGISTRY = {
    "sst2": {
        "hf_name":     "stanfordnlp/sst2",
        "text_col":    "sentence",
        "train_split": "train",
        "val_split":   "validation",
    },
    "imdb": {
        "hf_name":     "imdb",
        "text_col":    "text",
        "train_split": "train",
        "val_split":   "test",
    },
    "yelp_polarity": {
        "hf_name":     "yelp_polarity",
        "text_col":    "text",
        "train_split": "train",
        "val_split":   "test",
    },
}

_cache: dict = {}   # key: (dataset_id, split_name)

_SUBSAMPLE_SEED = 42


def list_datasets() -> List[str]:
    """Return names of all registered datasets."""
    return list(_DATASET_REGISTRY.keys())


def _load_split(dataset_id: str, split: str):
    key = (dataset_id, split)
    if key not in _cache:
        cfg = _DATASET_REGISTRY[dataset_id]
        hf_split = cfg[split + "_split"]
        _cache[key] = load_dataset(cfg["hf_name"], split=hf_split)
    return _cache[(dataset_id, split)]


def get_train_sentences(
    dataset_id: str = "sst2",
) -> Tuple[List[str], List[int]]:
    """Returns (sentences, labels) from the training split."""
    cfg = _DATASET_REGISTRY[dataset_id]
    ds = _load_split(dataset_id, "train")
    return ds[cfg["text_col"]], ds["label"]


def get_val_sentences(
    dataset_id: str = "sst2",
    max_examples: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    """
    Returns (sentences, labels) from the validation/test split.

    max_examples: if set, randomly subsample to this many examples using a
    fixed seed so results are reproducible. Useful to keep IMDb/Yelp
    (25-38k examples) comparable in size to SST-2 (872 examples).
    """
    cfg = _DATASET_REGISTRY[dataset_id]
    ds = _load_split(dataset_id, "val")
    sentences = list(ds[cfg["text_col"]])
    labels = list(ds["label"])

    if max_examples is not None and len(sentences) > max_examples:
        rng = _random.Random(_SUBSAMPLE_SEED)
        indices = rng.sample(range(len(sentences)), max_examples)
        indices.sort()
        sentences = [sentences[i] for i in indices]
        labels = [labels[i] for i in indices]

    return sentences, labels


if __name__ == "__main__":
    for ds_id in list_datasets():
        train_sents, train_labels = get_train_sentences(ds_id)
        val_sents, val_labels = get_val_sentences(ds_id, max_examples=872)
        neg = sum(1 for l in val_labels if l == 0)
        pos = sum(1 for l in val_labels if l == 1)
        print(f"\n{ds_id}")
        print(f"  Train: {len(train_sents):,}")
        print(f"  Val  : {len(val_sents):,}  (neg={neg}, pos={pos})")
        print(f"  Sample: {val_sents[0][:80]!r}")
