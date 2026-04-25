"""
load_data.py
Loads the Stanford Sentiment Treebank (SST-2) dataset from HuggingFace.

Note: The test split labels are hidden (-1), so all evaluation is done on
the validation split (872 examples).
"""

from datasets import load_dataset
from typing import Tuple, List


_DATASET_NAME = "stanfordnlp/sst2"
_cache: dict = {}


def _load_split(split: str):
    if split not in _cache:
        ds = load_dataset(_DATASET_NAME, split=split)
        _cache[split] = ds
    return _cache[split]


def get_train_sentences() -> Tuple[List[str], List[int]]:
    """Returns (sentences, labels) from the training split (~67k examples)."""
    ds = _load_split("train")
    return ds["sentence"], ds["label"]


def get_val_sentences() -> Tuple[List[str], List[int]]:
    """Returns (sentences, labels) from the validation split (872 examples)."""
    ds = _load_split("validation")
    return ds["sentence"], ds["label"]


if __name__ == "__main__":
    train_sents, train_labels = get_train_sentences()
    val_sents, val_labels = get_val_sentences()

    print(f"Train size : {len(train_sents):,}")
    print(f"Val size   : {len(val_sents):,}")
    print(f"Label dist (val) — 0: {val_labels.count(0)}, 1: {val_labels.count(1)}")
    print("\nSample validation sentences:")
    for i in range(3):
        print(f"  [{val_labels[i]}] {val_sents[i]!r}")
