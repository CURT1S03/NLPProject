"""
baseline_nbow.py
Neural Bag-of-Words (NBOW) model using GloVe-100d sentence embeddings.

Architecture
------------
  sentence → average of GloVe-100d token vectors → MLP (100→64→2) → softmax

GloVe
-----
  Download glove.6B.100d.txt from https://nlp.stanford.edu/data/glove.6B.zip
  and place it at: <project_root>/data/glove.6B.100d.txt
  (or set GLOVE_PATH in environment)

  The loaded embedding matrix is cached as a numpy .npz file alongside the
  GloVe text file to speed up repeated runs.

Usage
-----
  from models.baseline_nbow import NBOWModel

  model = NBOWModel()
  model.train()
  preds = model.predict(sentences)   # list[str] -> np.ndarray of 0/1
"""

import os
import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
GLOVE_PATH = os.environ.get(
    "GLOVE_PATH",
    os.path.join(_PROJECT_ROOT, "data", "glove.6B.100d.txt"),
)
_GLOVE_CACHE = GLOVE_PATH.replace(".txt", "_cache.npz")
_MODEL_CACHE = os.path.join(_PROJECT_ROOT, "results", "nbow_model.pt")

EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_CLASSES = 2
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
SEED = 42


# ---------------------------------------------------------------------------
# GloVe loader
# ---------------------------------------------------------------------------

def _load_glove(path: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Returns (word2idx, embedding_matrix).
    word2idx maps a word to its row index in embedding_matrix (shape [V, 100]).
    Index 0 is reserved as an unknown/padding zero vector.
    """
    cache = path.replace(".txt", "_cache.npz")
    if os.path.exists(cache):
        print(f"[NBOW] Loading GloVe cache from {cache}")
        data = np.load(cache, allow_pickle=True)
        words = data["words"].tolist()
        vecs = data["vecs"]
        word2idx = {w: i + 1 for i, w in enumerate(words)}
        pad = np.zeros((1, EMBED_DIM), dtype=np.float32)
        embedding_matrix = np.vstack([pad, vecs])
        return word2idx, embedding_matrix

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GloVe file not found at {path}.\n"
            "Download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip\n"
            "and extract glove.6B.100d.txt into the data/ directory."
        )

    print(f"[NBOW] Loading GloVe from {path} (first run, will cache) ...")
    words, vecs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vecs.append(np.array(parts[1:], dtype=np.float32))

    vecs_np = np.stack(vecs)
    np.savez_compressed(cache, words=np.array(words), vecs=vecs_np)
    print(f"[NBOW] GloVe cache saved to {cache}")

    word2idx = {w: i + 1 for i, w in enumerate(words)}
    pad = np.zeros((1, EMBED_DIM), dtype=np.float32)
    embedding_matrix = np.vstack([pad, vecs_np])
    return word2idx, embedding_matrix


# ---------------------------------------------------------------------------
# Sentence → embedding
# ---------------------------------------------------------------------------

def sentences_to_embeddings(
    sentences: list[str],
    word2idx: dict[str, np.ndarray],
    embedding_matrix: np.ndarray,
) -> np.ndarray:
    """Average GloVe vectors for each sentence. OOV tokens → zero vector."""
    result = np.zeros((len(sentences), EMBED_DIM), dtype=np.float32)
    for i, sent in enumerate(sentences):
        raw = sent.lower().split()
        tokens = [t.strip(string.punctuation) for t in raw]
        tokens = [t for t in tokens if t]
        indices = [word2idx[t] for t in tokens if t in word2idx]
        if indices:
            result[i] = embedding_matrix[indices].mean(axis=0)
    return result


# ---------------------------------------------------------------------------
# MLP definition
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

class NBOWModel:

    def __init__(
        self,
        glove_path: str = GLOVE_PATH,
        model_cache: str = _MODEL_CACHE,
    ):
        self.glove_path = glove_path
        self.model_cache = model_cache
        self.word2idx: dict | None = None
        self.embedding_matrix: np.ndarray | None = None
        self.mlp: _MLP | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_glove(self) -> None:
        if self.word2idx is None:
            self.word2idx, self.embedding_matrix = _load_glove(self.glove_path)

    def train(self, force_retrain: bool = False) -> None:
        """
        Train the MLP on GloVe-averaged sentence embeddings from SST-2 train split.
        Loads from cache if a saved checkpoint exists (unless force_retrain=True).
        """
        self._ensure_glove()

        if not force_retrain and os.path.exists(self.model_cache):
            print(f"[NBOW] Loading cached model from {self.model_cache}")
            self.mlp = _MLP().to(self.device)
            self.mlp.load_state_dict(torch.load(self.model_cache, map_location=self.device, weights_only=True))
            self.mlp.eval()
            return

        from data.load_data import get_train_sentences
        print(f"[NBOW] Training MLP on {self.device} ...")

        torch.manual_seed(SEED)
        train_sents, train_labels = get_train_sentences()

        X_train = sentences_to_embeddings(train_sents, self.word2idx, self.embedding_matrix)
        y_train = np.array(train_labels, dtype=np.int64)

        X_t = torch.tensor(X_train)
        y_t = torch.tensor(y_train)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.mlp = _MLP().to(self.device)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        self.mlp.train()
        for epoch in range(1, EPOCHS + 1):
            total_loss = 0.0
            correct = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.mlp(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
            avg_loss = total_loss / len(y_train)
            acc = correct / len(y_train)
            print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  train_acc={acc:.4f}")

        self.mlp.eval()
        os.makedirs(os.path.dirname(self.model_cache), exist_ok=True)
        torch.save(self.mlp.state_dict(), self.model_cache)
        print(f"[NBOW] Model saved to {self.model_cache}")

    def predict(self, sentences: list[str]) -> np.ndarray:
        if self.mlp is None:
            raise RuntimeError("Model is not trained. Call .train() first.")
        self._ensure_glove()

        X = sentences_to_embeddings(sentences, self.word2idx, self.embedding_matrix)
        X_t = torch.tensor(X).to(self.device)

        self.mlp.eval()
        with torch.no_grad():
            logits = self.mlp(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.load_data import get_val_sentences

    model = NBOWModel()
    model.train()

    val_sents, val_labels = get_val_sentences()
    preds = model.predict(val_sents)
    acc = accuracy_score(val_labels, preds)
    print(f"[NBOW] Clean validation accuracy: {acc:.4f}")
