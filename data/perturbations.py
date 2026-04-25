"""
perturbations.py
Applies controlled grammatical perturbations to a list of sentences.

Perturbation types
------------------
  punct_removal   : Remove all punctuation characters (severity ignored — always full removal).
  spelling_errors : Randomly corrupt a fraction of tokens with character-level noise.
  word_deletion   : Randomly drop a fraction of tokens.
  word_order      : Randomly swap adjacent token pairs with a given probability.

Severity
--------
  For `punct_removal`, severity is ignored.
  For all others, severity ∈ {0.1, 0.3, 0.5} controls the fraction of tokens affected.

Usage
-----
  from data.perturbations import perturb
  noisy = perturb(sentences, ptype="spelling_errors", severity=0.3)
"""

import random
import string
from typing import List, Optional

SEED = 42

PERTURBATION_TYPES = ["punct_removal", "spelling_errors", "word_deletion", "word_order"]
SEVERITIES = [0.1, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_punct(sentence: str) -> str:
    return sentence.translate(str.maketrans("", "", string.punctuation))


def _corrupt_word(word: str, rng: random.Random) -> str:
    """Apply one random character-level operation to a word."""
    if len(word) == 0:
        return word
    op = rng.choice(["swap", "delete", "insert", "transpose"])

    if op == "swap" and len(word) >= 2:
        # Replace a random char with a random lowercase letter
        idx = rng.randrange(len(word))
        replacement = rng.choice(string.ascii_lowercase)
        return word[:idx] + replacement + word[idx + 1:]

    elif op == "delete" and len(word) >= 2:
        idx = rng.randrange(len(word))
        return word[:idx] + word[idx + 1:]

    elif op == "insert":
        idx = rng.randrange(len(word) + 1)
        char = rng.choice(string.ascii_lowercase)
        return word[:idx] + char + word[idx:]

    elif op == "transpose" and len(word) >= 2:
        # Swap two adjacent characters
        idx = rng.randrange(len(word) - 1)
        lst = list(word)
        lst[idx], lst[idx + 1] = lst[idx + 1], lst[idx]
        return "".join(lst)

    # Fallback: replace a character
    idx = rng.randrange(len(word))
    return word[:idx] + rng.choice(string.ascii_lowercase) + word[idx + 1:]


def _apply_spelling_errors(sentence: str, severity: float, rng: random.Random) -> str:
    tokens = sentence.split()
    if not tokens:
        return sentence
    result = []
    for token in tokens:
        if rng.random() < severity:
            # Only corrupt alphanumeric content; preserve pure-punct tokens
            has_alpha = any(c.isalpha() for c in token)
            result.append(_corrupt_word(token, rng) if has_alpha else token)
        else:
            result.append(token)
    return " ".join(result)


def _apply_word_deletion(sentence: str, severity: float, rng: random.Random) -> str:
    tokens = sentence.split()
    if len(tokens) <= 1:
        return sentence  # never produce empty sentences
    kept = [t for t in tokens if rng.random() > severity]
    # Guarantee at least one token survives
    if not kept:
        kept = [rng.choice(tokens)]
    return " ".join(kept)


def _apply_word_order(sentence: str, severity: float, rng: random.Random) -> str:
    tokens = sentence.split()
    if len(tokens) <= 1:
        return sentence
    tokens = tokens[:]
    i = 0
    while i < len(tokens) - 1:
        if rng.random() < severity:
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
            i += 2  # skip the already-swapped pair to avoid cascading
        else:
            i += 1
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def perturb(
    sentences: List[str],
    ptype: str,
    severity: Optional[float] = None,
    seed: int = SEED,
) -> List[str]:
    """
    Apply a perturbation to every sentence in the list.

    Parameters
    ----------
    sentences : list of str
    ptype     : one of PERTURBATION_TYPES
    severity  : float in [0, 1]; ignored for 'punct_removal'
    seed      : random seed for reproducibility

    Returns
    -------
    list of str — perturbed sentences (same length as input)
    """
    if ptype not in PERTURBATION_TYPES:
        raise ValueError(f"Unknown perturbation type {ptype!r}. Choose from {PERTURBATION_TYPES}.")

    if ptype != "punct_removal" and severity is None:
        raise ValueError(f"severity must be specified for ptype={ptype!r}.")

    rng = random.Random(seed)

    if ptype == "punct_removal":
        return [_remove_punct(s) for s in sentences]

    elif ptype == "spelling_errors":
        return [_apply_spelling_errors(s, severity, rng) for s in sentences]

    elif ptype == "word_deletion":
        return [_apply_word_deletion(s, severity, rng) for s in sentences]

    elif ptype == "word_order":
        return [_apply_word_order(s, severity, rng) for s in sentences]


# ---------------------------------------------------------------------------
# Verification / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "The movie was surprisingly moving and well-acted.",
        "A dull, predictable film with no redeeming qualities.",
        "Absolutely fantastic performances all around!",
    ]

    print("=" * 60)
    print("PERTURBATION VERIFICATION")
    print("=" * 60)

    for ptype in PERTURBATION_TYPES:
        print(f"\n--- {ptype} ---")
        if ptype == "punct_removal":
            result = perturb(samples, ptype)
            for orig, pert in zip(samples, result):
                print(f"  ORIG : {orig}")
                print(f"  PERT : {pert}")
                print()
        else:
            for severity in SEVERITIES:
                result = perturb(samples, ptype, severity)
                print(f"  severity={severity}")
                for orig, pert in zip(samples, result):
                    print(f"    ORIG : {orig}")
                    print(f"    PERT : {pert}")
                print()
