"""
baseline_lr.py
Logistic Regression + TF-IDF baseline for SST-2 sentiment classification.

Trains on the full SST-2 training split (~67k sentences) and caches the
fitted model to disk so subsequent runs skip re-training.

Usage
-----
  from models.baseline_lr import LogisticRegressionModel

  model = LogisticRegressionModel()
  model.train()                     # loads from cache if available
  preds = model.predict(sentences)  # list[str] -> np.ndarray of 0/1
"""

import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

_MODEL_CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "lr_model.joblib")


class LogisticRegressionModel:

    def __init__(self, cache_path: str = _MODEL_CACHE):
        self.cache_path = cache_path
        self.pipeline: Pipeline | None = None

    def train(self, force_retrain: bool = False) -> None:
        """
        Fit the TF-IDF + LR pipeline on SST-2 training data.
        Loads from disk if a cached model exists (unless force_retrain=True).
        """
        if not force_retrain and os.path.exists(self.cache_path):
            print(f"[LR] Loading cached model from {self.cache_path}")
            self.pipeline = joblib.load(self.cache_path)
            return

        from data.load_data import get_train_sentences
        print("[LR] Training TF-IDF + Logistic Regression ...")
        train_sents, train_labels = get_train_sentences()

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50_000,
                sublinear_tf=True,
            )),
            ("lr", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        self.pipeline.fit(train_sents, train_labels)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        joblib.dump(self.pipeline, self.cache_path)
        print(f"[LR] Model saved to {self.cache_path}")

    def predict(self, sentences: list[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained. Call .train() first.")
        return self.pipeline.predict(sentences)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.load_data import get_val_sentences

    model = LogisticRegressionModel()
    model.train()

    val_sents, val_labels = get_val_sentences()
    preds = model.predict(val_sents)
    acc = accuracy_score(val_labels, preds)
    print(f"[LR] Clean validation accuracy: {acc:.4f}")
