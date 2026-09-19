"""
corpus_utils.py
----------------
Shared utilities for building the WordNet-based Hangman vocabulary and the
letter-statistics tables used as ML features. Imported by both the training
notebook and app.py so the two stay perfectly in sync.
"""

import os
import string
import pandas as pd

MIN_LEN = 4
MAX_LEN = 15
ALPHABET = list(string.ascii_lowercase)


def ensure_wordnet():
    """Download WordNet (and the multilingual index it needs) if missing."""
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


def build_vocab_from_wordnet():
    """Extract & filter the Hangman vocabulary directly from NLTK WordNet."""
    ensure_wordnet()
    from nltk.corpus import wordnet as wn

    words = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if word.isalpha():
                words.add(word)

    total_raw = len(words)

    filtered = set()
    for w in words:
        if not w.isalpha():
            continue
        if not w.isascii():
            continue
        if "-" in w or "_" in w or " " in w:
            continue
        if any(ch.isdigit() for ch in w):
            continue
        if MIN_LEN <= len(w) <= MAX_LEN:
            filtered.add(w)

    vocab = sorted(filtered)
    return vocab, total_raw


def save_vocab_csv(vocab, path="wordnet_corpus.csv"):
    """Save the vocabulary list as a CSV file."""
    df = pd.DataFrame({"word": vocab, "length": [len(w) for w in vocab]})
    df.to_csv(path, index=False)
    return path


def load_or_build_vocab(csv_path="wordnet_corpus.csv"):
    """Load the cached CSV if present, otherwise regenerate from WordNet."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, keep_default_na=False, na_filter=False, dtype=str)
        vocab = sorted(df["word"].astype(str).tolist())
        return vocab
    vocab, _ = build_vocab_from_wordnet()
    save_vocab_csv(vocab, csv_path)
    return vocab


def build_letter_stats(vocab):
    """
    Compute frequency tables used as ML features:
      - global_freq[letter]                 -> P(letter in a random word)
      - length_freq[length][letter]         -> P(letter in word | word length)
      - position_freq[(length,pos)][letter] -> P(letter at position | length)
      - pair_freq[letter1][letter2]         -> P(letter2 in word | letter1 in word)
    """
    global_count = {c: 0 for c in ALPHABET}
    length_count = {}
    position_count = {}
    pair_count = {c: {c2: 0 for c2 in ALPHABET} for c in ALPHABET}
    length_totals = {}

    n_words = len(vocab)

    for w in vocab:
        L = len(w)
        length_totals[L] = length_totals.get(L, 0) + 1
        letters_in_word = set(w)

        for c in letters_in_word:
            global_count[c] += 1

        length_count.setdefault(L, {c: 0 for c in ALPHABET})
        for c in letters_in_word:
            length_count[L][c] += 1

        for pos, ch in enumerate(w):
            key = (L, pos)
            position_count.setdefault(key, {c: 0 for c in ALPHABET})
            position_count[key][ch] += 1

        for c1 in letters_in_word:
            for c2 in letters_in_word:
                if c1 != c2:
                    pair_count[c1][c2] += 1

    global_freq = {c: global_count[c] / n_words for c in ALPHABET}

    length_freq = {}
    for L, counts in length_count.items():
        tot = length_totals[L]
        length_freq[L] = {c: counts[c] / tot for c in ALPHABET}

    position_freq = {}
    for key, counts in position_count.items():
        L, pos = key
        tot = length_totals[L]
        position_freq[key] = {c: counts[c] / tot for c in ALPHABET}

    pair_freq = {}
    for c1 in ALPHABET:
        denom = global_count[c1] if global_count[c1] > 0 else 1
        pair_freq[c1] = {c2: pair_count[c1][c2] / denom for c2 in ALPHABET}

    stats = {
        "global_freq": global_freq,
        "length_freq": length_freq,
        "position_freq": position_freq,
        "pair_freq": pair_freq,
        "length_totals": length_totals,
        "n_words": n_words,
    }
    return stats
