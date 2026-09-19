"""
WordNet AI Hangman — Production Gradio Application
===================================================
A real machine-learning powered Hangman game where:
- The secret word vocabulary is sourced directly from NLTK WordNet (lengths 4-15).
- The AI letter recommender uses a trained scikit-learn HistGradientBoostingClassifier.
- Feature engineering accounts for global letter frequency, length-conditioned frequency,
  positional letter distributions across blanks, and character co-occurrence patterns.
- Guessed and revealed letters are strictly excluded from AI recommendations.
- Zero retraining on app startup: loads pre-trained model artifacts directly.
"""

import os
import json
import pickle
import random
import string
import numpy as np
import gradio as gr

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
ALPHABET = list(string.ascii_lowercase)
MAX_WRONG = 6

# Base directory of the application script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Search paths for model artifacts (checks model/ first, then artifacts/)
def find_file(relative_options):
    for path in relative_options:
        full_path = os.path.join(BASE_DIR, path)
        if os.path.exists(full_path):
            return full_path
        if os.path.exists(path):
            return path
    return os.path.join(BASE_DIR, relative_options[0])

MODEL_PATH = find_file(["model/model.pkl", "artifacts/model.pkl"])
STATS_PATH = find_file(["model/stats.pkl", "artifacts/stats.pkl"])
VOCAB_PATH = find_file(["model/wordnet_corpus.csv", "wordnet_corpus.csv", "artifacts/wordnet_corpus.csv"])
CONFIG_PATH = find_file(["model/config.json", "config.json"])

HANGMAN_STAGES = [
    """
      +---+
      |   |
          |
          |
          |
          |
    =========""",
    """
      +---+
      |   |
      O   |
          |
          |
          |
    =========""",
    """
      +---+
      |   |
      O   |
      |   |
          |
          |
    =========""",
    """
      +---+
      |   |
      O   |
     /|   |
          |
          |
    =========""",
    """
      +---+
      |   |
      O   |
     /|\\  |
          |
          |
    =========""",
    """
      +---+
      |   |
      O   |
     /|\\  |
     /    |
          |
    =========""",
    """
      +---+
      |   |
      O   |
     /|\\  |
     / \\  |
          |
    =========""",
]

DIFFICULTY_RANGES = {
    "Easy (4–6 letters)": (4, 6),
    "Medium (7–9 letters)": (7, 9),
    "Hard (10–15 letters)": (10, 15),
    "Any Length (4–15 letters)": (4, 15),
}

# ---------------------------------------------------------------------------
# Self-contained Feature Engineering & WordNet Loading
# ---------------------------------------------------------------------------
def ensure_wordnet():
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

def build_vocab_from_wordnet():
    ensure_wordnet()
    from nltk.corpus import wordnet as wn
    words = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if word.isalpha() and word.isascii() and 4 <= len(word) <= 15:
                words.add(word)
    return sorted(words)

def load_vocabulary(path):
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path, keep_default_na=False, na_filter=False, dtype=str)
        return sorted(df["word"].astype(str).tolist())
    vocab = build_vocab_from_wordnet()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import pandas as pd
    pd.DataFrame({"word": vocab, "length": [len(w) for w in vocab]}).to_csv(path, index=False)
    return vocab

def compute_letter_stats(vocab):
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
    length_freq = {L: {c: counts[c] / length_totals[L] for c in ALPHABET} for L, counts in length_count.items()}
    position_freq = {k: {c: counts[c] / length_totals[k[0]] for c in ALPHABET} for k, counts in position_count.items()}
    pair_freq = {c1: {c2: pair_count[c1][c2] / max(global_count[c1], 1) for c2 in ALPHABET} for c1 in ALPHABET}

    return {
        "global_freq": global_freq,
        "length_freq": length_freq,
        "position_freq": position_freq,
        "pair_freq": pair_freq,
        "length_totals": length_totals,
        "n_words": n_words,
    }

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
        pos_scores = [position_freq.get((word_length, pos), {}).get(candidate, 0.0) for pos in blank_positions]
        f_position_avg = sum(pos_scores) / len(pos_scores)
        f_position_max = max(pos_scores)
    else:
        f_position_avg = 0.0
        f_position_max = 0.0

    revealed_letters = set(c for c in pattern if c != "_")
    if revealed_letters:
        pair_scores = [pair_freq.get(r, {}).get(candidate, 0.0) for r in revealed_letters]
        f_pair_avg = sum(pair_scores) / len(pair_scores)
    else:
        f_pair_avg = f_global

    wrong_letters = guessed_letters - revealed_letters
    if wrong_letters:
        wrong_scores = [pair_freq.get(w, {}).get(candidate, 0.0) for w in wrong_letters]
        f_wrong_pair_avg = sum(wrong_scores) / len(wrong_scores)
    else:
        f_wrong_pair_avg = f_global

    progress = n_revealed / max(word_length, 1)

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

# ---------------------------------------------------------------------------
# App Initialization & Artifact Loading (Zero Startup Retraining)
# ---------------------------------------------------------------------------
print(f"Loading vocabulary from {VOCAB_PATH}...")
VOCAB = load_vocabulary(VOCAB_PATH)
print(f"Loaded {len(VOCAB)} words from WordNet vocabulary.")

if os.path.exists(STATS_PATH):
    with open(STATS_PATH, "rb") as f:
        STATS = pickle.load(f)
    print(f"Loaded frequency statistics from {STATS_PATH}.")
else:
    print(f"Computing frequency statistics from vocabulary...")
    STATS = compute_letter_stats(VOCAB)
    os.makedirs(os.path.dirname(STATS_PATH) or ".", exist_ok=True)
    with open(STATS_PATH, "wb") as f:
        pickle.dump(STATS, f)

MODEL = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    print(f"Loaded trained ML model from {MODEL_PATH}.")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Using frequency baseline.")

VOCAB_BY_DIFFICULTY = {
    label: [w for w in VOCAB if lo <= len(w) <= hi]
    for label, (lo, hi) in DIFFICULTY_RANGES.items()
}

# ---------------------------------------------------------------------------
# ML Inference / Recommendation Engine
# ---------------------------------------------------------------------------
def rank_letters(word_length, pattern, guessed_letters, top_k=5):
    """
    Score unguessed, unrevealed letters using the trained model.
    Guarantees:
      - Letters already guessed are excluded.
      - Letters already revealed are excluded.
      - Returns top_k letter predictions with relative percentage confidence.
    """
    revealed = set(c for c in pattern if c != "_")
    excluded = set(guessed_letters) | revealed
    remaining = [c for c in ALPHABET if c not in excluded]

    if not remaining:
        return [], None, 0.0

    if MODEL is not None:
        feats = np.array([
            make_features(word_length, pattern, guessed_letters, c, STATS)
            for c in remaining
        ])
        raw_probs = MODEL.predict_proba(feats)[:, 1]
    else:
        length_freq = STATS["length_freq"].get(word_length, STATS["global_freq"])
        raw_probs = np.array([length_freq.get(c, STATS["global_freq"].get(c, 0.01)) for c in remaining])

    # Calculate normalized relative confidence distribution among valid remaining letters
    prob_sum = raw_probs.sum()
    if prob_sum > 0:
        confidences = (raw_probs / prob_sum) * 100.0
    else:
        confidences = np.full(len(remaining), 100.0 / len(remaining))

    ranked = sorted(
        zip(remaining, confidences, raw_probs),
        key=lambda item: -item[1]
    )

    top_letter = ranked[0][0]
    top_confidence = ranked[0][1]

    return ranked[:top_k], top_letter, top_confidence

# ---------------------------------------------------------------------------
# Game State Logic
# ---------------------------------------------------------------------------
def init_game(difficulty="Medium (7–9 letters)"):
    pool = VOCAB_BY_DIFFICULTY.get(difficulty) or VOCAB
    secret = random.choice(pool)
    state = {
        "secret": secret,
        "pattern": ["_"] * len(secret),
        "guessed": set(),
        "wrong_count": 0,
        "game_over": False,
        "won": False,
        "message": "Game started! Make your first guess.",
        "difficulty": difficulty,
    }
    return state

def format_word_display(state):
    return " ".join(state["pattern"]).upper()

def format_guessed_display(state):
    if not state["guessed"]:
        return "None"
    return ", ".join(sorted(c.upper() for c in state["guessed"]))

def format_attempts_display(state):
    remaining = MAX_WRONG - state["wrong_count"]
    return f"{remaining} / {MAX_WRONG}"

def format_status_display(state):
    if state["game_over"]:
        if state["won"]:
            return f"🎉 WIN! You solved the word: {state['secret'].upper()}"
        else:
            return f"💀 GAME OVER! The word was: {state['secret'].upper()}"
    return state.get("message", "Keep playing!")

def format_ai_recommendation(top_letter, confidence):
    if not top_letter:
        return "—", "—"
    return top_letter.upper(), f"{confidence:.1f}%"

def format_top5_markdown(ranked):
    if not ranked:
        return "*(No unguessed letters remaining)*"
    lines = []
    for letter, conf, _ in ranked:
        lines.append(f"**{letter.upper()}** — {conf:.1f}%")
    return "\n\n".join(lines)

def apply_guess(state, letter):
    if state is None:
        state = init_game()

    if state["game_over"]:
        return state, "Game over! Click 'New Game' to play again."

    letter = (letter or "").strip().lower()
    if not letter or len(letter) != 1 or letter not in string.ascii_lowercase:
        return state, "⚠️ Please enter a single letter from A to Z."

    if letter in state["guessed"]:
        return state, f"⚠️ You already guessed '{letter.upper()}'. Try a different letter."

    state["guessed"].add(letter)
    secret = state["secret"]

    if letter in secret:
        for i, ch in enumerate(secret):
            if ch == letter:
                state["pattern"][i] = letter
        if "_" not in state["pattern"]:
            state["game_over"] = True
            state["won"] = True
            state["message"] = f"🎉 WIN! You guessed the word: {secret.upper()}!"
        else:
            state["message"] = f"✅ Great guess! '{letter.upper()}' is in the word."
    else:
        state["wrong_count"] += 1
        if state["wrong_count"] >= MAX_WRONG:
            state["game_over"] = True
            state["won"] = False
            state["message"] = f"💀 GAME OVER! Out of attempts. The word was '{secret.upper()}'."
        else:
            remaining = MAX_WRONG - state["wrong_count"]
            state["message"] = f"❌ '{letter.upper()}' is not in the word. {remaining} attempts left."

    return state, state["message"]

def render_state(state):
    word_disp = format_word_display(state)
    guessed_disp = format_guessed_display(state)
    attempts_disp = format_attempts_display(state)
    ascii_art = HANGMAN_STAGES[state["wrong_count"]]
    status_disp = format_status_display(state)

    if state["game_over"]:
        top_letter_disp = "—"
        conf_disp = "—"
        top5_md = "Game finished. Start a New Game to see recommendations."
    else:
        ranked, top_letter, conf = rank_letters(
            len(state["secret"]),
            state["pattern"],
            state["guessed"],
            top_k=5
        )
        top_letter_disp, conf_disp = format_ai_recommendation(top_letter, conf)
        top5_md = format_top5_markdown(ranked)

    return (
        word_disp,
        guessed_disp,
        attempts_disp,
        ascii_art,
        status_disp,
        top_letter_disp,
        conf_disp,
        top5_md,
        state,
        "",  # clear input box
    )

# ---------------------------------------------------------------------------
# Gradio Callbacks
# ---------------------------------------------------------------------------
def cb_new_game(difficulty):
    state = init_game(difficulty)
    return render_state(state)

def cb_guess(letter_text, state):
    state, _ = apply_guess(state, letter_text)
    return render_state(state)

def cb_key_press(letter, state):
    return cb_guess(letter, state)

def cb_use_ai_recommendation(state):
    if state is None or state["game_over"]:
        return render_state(state)
    ranked, top_letter, _ = rank_letters(
        len(state["secret"]),
        state["pattern"],
        state["guessed"],
        top_k=1
    )
    if top_letter:
        return cb_guess(top_letter, state)
    return render_state(state)

# ---------------------------------------------------------------------------
# UI Theme & Layout
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* App Theme & Typography */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --bg-card: rgba(30, 41, 59, 0.7);
    --border-color: rgba(255, 255, 255, 0.1);
}

body, .gradio-container {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.title-header {
    text-align: center;
    padding: 1.2rem 0;
    margin-bottom: 0.5rem;
}

.title-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #6366f1, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.title-header p {
    color: #94a3b8;
    font-size: 1.05rem;
}

.word-display textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.6rem !important;
    text-align: center !important;
    color: #38bdf8 !important;
    background: rgba(15, 23, 42, 0.85) !important;
    border: 2px solid #38bdf844 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.15);
}

.ascii-gallows textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.15rem !important;
    line-height: 1.25 !important;
    color: #f59e0b !important;
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

.ai-card {
    background: linear-gradient(145deg, rgba(99, 102, 241, 0.12), rgba(168, 85, 247, 0.12));
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
}

.ai-top-letter textarea {
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    text-align: center !important;
    color: #a855f7 !important;
    background: rgba(15, 23, 42, 0.7) !important;
    border-radius: 12px !important;
}

.ai-confidence textarea {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    color: #10b981 !important;
    background: rgba(15, 23, 42, 0.7) !important;
}

.kbd-btn {
    min-width: 38px !important;
    height: 40px !important;
    padding: 0 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

.kbd-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.action-btn-primary {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
}

.action-btn-secondary {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
}
"""

with gr.Blocks(title="🎮 WordNet AI Hangman") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    gr.HTML(
        """
        <div class="title-header">
            <h1>🎮 WordNet AI Hangman</h1>
            <p>A supervised machine learning Hangman game powered by <b>NLTK WordNet</b> and a trained gradient-boosting model.</p>
        </div>
        """
    )

    game_state = gr.State(value=None)

    # Top Control Bar
    with gr.Row():
        difficulty = gr.Dropdown(
            choices=list(DIFFICULTY_RANGES.keys()),
            value="Medium (7–9 letters)",
            label="Difficulty (Word Length)",
            scale=3,
        )
        new_game_btn = gr.Button("🔄 New Game", variant="primary", scale=1, elem_classes=["action-btn-primary"])

    # Main Board Layout
    with gr.Row():
        # Left: Hangman & Game Board
        with gr.Column(scale=3):
            word_box = gr.Textbox(
                label="Word",
                value="",
                interactive=False,
                lines=1,
                elem_classes=["word-display"],
            )

            with gr.Row():
                guessed_box = gr.Textbox(
                    label="Guessed Letters",
                    value="None",
                    interactive=False,
                    scale=2,
                )
                attempts_box = gr.Textbox(
                    label="Remaining Attempts",
                    value=f"{MAX_WRONG} / {MAX_WRONG}",
                    interactive=False,
                    scale=1,
                )

            status_box = gr.Textbox(
                label="Status",
                value="Keep playing!",
                interactive=False,
            )

            # Guess Input Controls
            with gr.Row():
                letter_input = gr.Textbox(
                    label="Enter Letter",
                    placeholder="e.g. E",
                    max_lines=1,
                    scale=3,
                )
                guess_btn = gr.Button(
                    "Guess Letter",
                    variant="primary",
                    scale=1,
                    elem_classes=["action-btn-secondary"],
                )

            # Interactive Virtual Keyboard
            gr.Markdown("##### 🔤 Virtual Keyboard")
            letter_buttons = {}
            row1 = ALPHABET[0:9]    # a - i
            row2 = ALPHABET[9:18]   # j - r
            row3 = ALPHABET[18:26]  # s - z

            for row in [row1, row2, row3]:
                with gr.Row():
                    for char in row:
                        btn = gr.Button(char.upper(), elem_classes=["kbd-btn"])
                        letter_buttons[char] = btn

        # Right: ASCII Gallows & AI Recommender
        with gr.Column(scale=2):
            ascii_box = gr.Textbox(
                label="Hangman Gallows",
                value=HANGMAN_STAGES[0],
                interactive=False,
                lines=8,
                elem_classes=["ascii-gallows"],
            )

            # AI Insights Card
            with gr.Group(elem_classes=["ai-card"]):
                gr.Markdown("### 🤖 AI Recommendation")
                with gr.Row():
                    ai_rec_letter = gr.Textbox(
                        label="Next Letter",
                        value="—",
                        interactive=False,
                        scale=1,
                        elem_classes=["ai-top-letter"],
                    )
                    ai_confidence = gr.Textbox(
                        label="Confidence",
                        value="—",
                        interactive=False,
                        scale=1,
                        elem_classes=["ai-confidence"],
                    )

                ai_pick_btn = gr.Button("✨ Use AI Recommendation", variant="secondary")

                gr.Markdown("#### 📊 Top 5 AI Predictions")
                top5_markdown = gr.Markdown("Start game to see AI predictions.")

    # Model Information Footer
    gr.Markdown(
        """
        ---
        ### ℹ️ How the AI Model Works
        * **Supervised Learning**: Powered by a scikit-learn `HistGradientBoostingClassifier` trained on ~1.35 million Hangman states.
        * **Feature Engineering**: Calculates global unigram probabilities, length-conditioned frequencies, blank position distributions, and letter co-occurrence with confirmed/rejected letters.
        * **Smart Filtering**: Automatically filters out all already-guessed and revealed letters so every recommendation is valid and actionable.
        """
    )

    # Output mapping
    outputs_list = [
        word_box,
        guessed_box,
        attempts_box,
        ascii_box,
        status_box,
        ai_rec_letter,
        ai_confidence,
        top5_markdown,
        game_state,
        letter_input,
    ]

    # Event Bindings
    new_game_btn.click(cb_new_game, inputs=[difficulty], outputs=outputs_list)
    demo.load(cb_new_game, inputs=[difficulty], outputs=outputs_list)

    guess_btn.click(cb_guess, inputs=[letter_input, game_state], outputs=outputs_list)
    letter_input.submit(cb_guess, inputs=[letter_input, game_state], outputs=outputs_list)
    ai_pick_btn.click(cb_use_ai_recommendation, inputs=[game_state], outputs=outputs_list)

    for char, btn in letter_buttons.items():
        btn.click(
            lambda state, c=char: cb_key_press(c, state),
            inputs=[game_state],
            outputs=outputs_list,
        )

# ---------------------------------------------------------------------------
# App Launch Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
