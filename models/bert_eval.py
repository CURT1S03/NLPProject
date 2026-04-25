"""
bert_eval.py
Zero-shot evaluation of DistilBERT fine-tuned on SST-2.

Uses the publicly available checkpoint:
    distilbert-base-uncased-finetuned-sst-2-english

No re-training is performed — this evaluates the model's inference-time
robustness to grammatical perturbations.

Usage
-----
  from models.bert_eval import DistilBERTModel

  model = DistilBERTModel()
  model.load()
  preds = model.predict(sentences)   # list[str] -> np.ndarray of 0/1
"""

import os
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score

_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
_BATCH_SIZE = 64
# Label mapping from the checkpoint's id2label — confirmed: NEGATIVE=0, POSITIVE=1
_LABEL_MAP = {"NEGATIVE": 0, "POSITIVE": 1}


class DistilBERTModel:

    def __init__(
        self,
        checkpoint: str = _CHECKPOINT,
        batch_size: int = _BATCH_SIZE,
    ):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self._pipe = None

    def load(self) -> None:
        """Load the HuggingFace text-classification pipeline (downloads once, then cached)."""
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"[BERT] Loading {self.checkpoint}  (device={'GPU' if device == 0 else 'CPU'}) ...")
        self._pipe = pipeline(
            "text-classification",
            model=self.checkpoint,
            device=device,
            truncation=True,
            max_length=128,
        )
        print("[BERT] Model loaded.")

    def predict(self, sentences: list[str]) -> np.ndarray:
        """
        Run inference on `sentences` in batches.
        Returns an int numpy array of 0/1 labels.
        """
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        all_preds = []
        # Run pipeline in explicit batches so we can show a progress bar
        for start in tqdm(
            range(0, len(sentences), self.batch_size),
            desc="[BERT] Inference",
            unit="batch",
        ):
            batch = sentences[start : start + self.batch_size]
            outputs = self._pipe(batch, truncation=True, max_length=128)
            for out in outputs:
                all_preds.append(_LABEL_MAP[out["label"]])
        return np.array(all_preds, dtype=np.int64)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.load_data import get_val_sentences

    model = DistilBERTModel()
    model.load()

    val_sents, val_labels = get_val_sentences()
    preds = model.predict(val_sents)
    acc = accuracy_score(val_labels, preds)
    print(f"[BERT] Clean validation accuracy: {acc:.4f}")
