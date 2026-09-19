
import random
import string

ALPHABET = list(string.ascii_lowercase)

def make_features(word_length, pattern, guessed_letters, candidate, stats):
    global_freq = stats["global_freq"]
    length_freq = stats["length_freq"].get(word_length, {})
    position_freq = stats["position_freq"]
    pair_freq = stats["pair_freq"]

    n_blanks = pattern.count("_")
    n_revealed = word_length - n_blanks
    n_guessed = len(guessed_letters)

    f_global = global_freq.get(candidate, 0.0)
    f_length = length_freq.get(candidate, 0.0)

    blank_positions = [i for i, c in enumerate(pattern) if c == "_"]

    if blank_positions:
        pos_scores = []

        for pos in blank_positions:
            key = (word_length, pos)
            pos_scores.append(
                position_freq.get(key, {}).get(candidate, 0.0)
            )

        f_position_avg = sum(pos_scores) / len(pos_scores)
        f_position_max = max(pos_scores)
    else:
        f_position_avg = 0.0
        f_position_max = 0.0

    revealed_letters = set(c for c in pattern if c != "_")

    if revealed_letters:
        pair_scores = [
            pair_freq.get(r, {}).get(candidate, 0.0)
            for r in revealed_letters
        ]
        f_pair_avg = sum(pair_scores) / len(pair_scores)
    else:
        f_pair_avg = f_global

    wrong_letters = guessed_letters - revealed_letters

    if wrong_letters:
        wrong_scores = [
            pair_freq.get(w, {}).get(candidate, 0.0)
            for w in wrong_letters
        ]
        f_wrong_pair_avg = sum(wrong_scores) / len(wrong_scores)
    else:
        f_wrong_pair_avg = f_global

    progress = n_revealed / word_length if word_length else 0.0

    return [
        f_global,
        f_length,
        f_position_avg,
        f_position_max,
        f_pair_avg,
        f_wrong_pair_avg,
        progress,
        n_blanks / max(word_length, 1),
        n_guessed / 26.0,
        word_length / 15.0,
    ]


FEATURE_NAMES = [
    "global_freq",
    "length_freq",
    "position_freq_avg",
    "position_freq_max",
    "pair_freq_with_revealed",
    "pair_freq_with_wrong",
    "progress",
    "blank_ratio",
    "guessed_ratio",
    "norm_word_length",
]

def simulate_game_state(word, rng):
    distinct_letters = list(set(word))
    rng.shuffle(distinct_letters)

    n_reveal = rng.randint(
        0,
        max(0, len(distinct_letters) - 1)
    )

    revealed = set(distinct_letters[:n_reveal])

    pattern = [
        ch if ch in revealed else "_"
        for ch in word
    ]

    not_in_word = [
        c for c in ALPHABET
        if c not in word
    ]

    rng.shuffle(not_in_word)

    n_wrong = rng.randint(
        0,
        min(5, len(not_in_word))
    )

    wrong_guessed = set(not_in_word[:n_wrong])

    guessed_letters = revealed | wrong_guessed

    return pattern, guessed_letters

def generate_training_data(
    vocab,
    stats,
    n_samples_per_word=3,
    candidates_per_state=6,
    seed=42
):
    rng = random.Random(seed)

    X, y = [], []

    for word in vocab:

        for _ in range(n_samples_per_word):

            pattern, guessed_letters = simulate_game_state(
                word, rng
            )

            if "_" not in pattern:
                continue

            remaining_letters = [
                c for c in ALPHABET
                if c not in guessed_letters
            ]

            if not remaining_letters:
                continue

            positive_candidates = [
                c for c in remaining_letters
                if c in word
            ]

            negative_candidates = [
                c for c in remaining_letters
                if c not in word
            ]

            rng.shuffle(positive_candidates)
            rng.shuffle(negative_candidates)

            n_pos = min(
                len(positive_candidates),
                max(1, candidates_per_state // 2)
            )

            n_neg = min(
                len(negative_candidates),
                candidates_per_state - n_pos
            )

            chosen = (
                [(c, 1) for c in positive_candidates[:n_pos]]
                + [(c, 0) for c in negative_candidates[:n_neg]]
            )

            for candidate, label in chosen:

                feats = make_features(
                    len(word),
                    pattern,
                    guessed_letters,
                    candidate,
                    stats
                )

                X.append(feats)
                y.append(label)

    return X, y
