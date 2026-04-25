import csv
import os
import random
import string
import sys
import re


SEED = 42
SEVERITIES = [0.1, 0.3, 0.5]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "perturbations")
OUTPUT_PATH = os.path.join(SAVE_DIR, "validation_perturbations.csv")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


FUNCTION_WORDS = set("""
a an the and or but if while although though because so
of in on at to for from with by as into onto about over under after before between through during
is am are was were be been being do does did have has had
will would can could should may might must
""".split())

NEGATION_WORDS = {"no", "nor", "not", "never"}


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans("", "", string.punctuation))

def normalize_sentence(sentence):
    sentence = sentence.strip()

    sentence = sentence.replace("``", '"')
    sentence = sentence.replace("''", '"')

    sentence = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|n't|'em)\b", r"\1", sentence)
    sentence = re.sub(r"\s+([,.;:!?%])", r"\1", sentence)

    sentence = re.sub(r"\(\s+", "(", sentence)
    sentence = re.sub(r"\s+\)", ")", sentence)

    sentence = re.sub(r'"\s+', '"', sentence)
    sentence = re.sub(r'\s+"', '"', sentence)

    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()

def clean_token(token):
    return token.strip(string.punctuation).lower()


def corrupt_word(word):
    if len(word) <= 1:
        return word

    op = random.choice(["swap", "delete", "insert", "transpose"])

    if op == "swap":
        chars = list(word)
        i = random.randrange(len(chars))
        j = random.randrange(len(chars))
        chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)

    if op == "delete":
        i = random.randrange(len(word))
        return word[:i] + word[i + 1:]

    if op == "insert":
        i = random.randrange(len(word) + 1)
        c = random.choice(string.ascii_lowercase)
        return word[:i] + c + word[i:]

    i = random.randrange(len(word) - 1)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def spelling_errors(sentence, severity):
    tokens = sentence.split()

    for i, token in enumerate(tokens):
        if any(ch.isalpha() for ch in token) and random.random() < severity:
            tokens[i] = corrupt_word(token)

    return " ".join(tokens)


def word_deletion(sentence, severity):
    tokens = sentence.split()

    if len(tokens) <= 1:
        return sentence

    candidates = []
    for i, token in enumerate(tokens):
        base = clean_token(token)
        if base in FUNCTION_WORDS and base not in NEGATION_WORDS:
            candidates.append(i)

    if not candidates:
        return sentence

    num_to_delete = round(len(tokens) * severity)
    num_to_delete = min(num_to_delete, len(candidates), len(tokens) - 1)

    if num_to_delete <= 0:
        return sentence

    delete_positions = set(random.sample(candidates, num_to_delete))
    kept = []

    for i, token in enumerate(tokens):
        if i not in delete_positions:
            kept.append(token)

    return " ".join(kept)


def word_order(sentence, severity):
    tokens = sentence.split()
    i = 0

    while i < len(tokens) - 1:
        if random.random() < severity:
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
            i += 2
        else:
            i += 1

    return " ".join(tokens)


def apply_steps(sentence, steps, severity):
    for step in steps:
        if step == "punct_removal":
            sentence = remove_punctuation(sentence)
        elif step == "spelling_errors":
            sentence = spelling_errors(sentence, severity)
        elif step == "word_deletion":
            sentence = word_deletion(sentence, severity)
        elif step == "word_order":
            sentence = word_order(sentence, severity)
        else:
            raise ValueError(f"Unknown perturbation type: {step}")

    return sentence


def make_conditions():
    conditions = [
        ("clean", None, []),
        ("punct_removal", None, ["punct_removal"]),
    ]

    for name in ["spelling_errors", "word_deletion", "word_order"]:
        for severity in SEVERITIES:
            conditions.append((name, severity, [name]))

    combinations = [
        ("punct_plus_spelling", ["punct_removal", "spelling_errors"]),
        ("deletion_plus_order", ["word_deletion", "word_order"]),
        ("spelling_plus_deletion", ["spelling_errors", "word_deletion"]),
        ("all_combined", ["punct_removal", "spelling_errors", "word_deletion", "word_order"]),
    ]

    for name, steps in combinations:
        for severity in SEVERITIES:
            conditions.append((name, severity, steps))

    return conditions


def save_validation_perturbations():
    from data.load_data import get_val_sentences

    sentences, labels = get_val_sentences()
    sentences = [normalize_sentence(sentence) for sentence in sentences]
    labels = list(labels)

    conditions = make_conditions()
    condition_outputs = []

    for perturbation_type, severity, steps in conditions:
        random.seed(SEED)
        severity_label = "N/A" if severity is None else f"{severity:.1f}"

        perturbed_sentences = []
        for sentence in sentences:
            perturbed_sentences.append(apply_steps(sentence, steps, severity))

        condition_outputs.append(
            (perturbation_type, severity_label, perturbed_sentences)
        )

    rows = []

    for i, sentence in enumerate(sentences):
        for perturbation_type, severity_label, perturbed_sentences in condition_outputs:
            rows.append({
                "sentence_index": i,
                "label": labels[i],
                "perturbation_type": perturbation_type,
                "severity": severity_label,
                "original_sentence": sentence,
                "perturbed_sentence": perturbed_sentences[i],
            })

    os.makedirs(SAVE_DIR, exist_ok=True)

    fieldnames = [
        "sentence_index",
        "label",
        "perturbation_type",
        "severity",
        "original_sentence",
        "perturbed_sentence",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    save_validation_perturbations()