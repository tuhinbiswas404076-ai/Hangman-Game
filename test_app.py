"""
test_app.py — Automated verification test suite for WordNet Hangman AI
"""

import os
import sys
import string

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def run_tests():
    print("==================================================")
    print("Testing WordNet Hangman AI Deployment Components")
    print("==================================================")
    checklist = []

    # 1. Test app.py imports successfully
    try:
        import app
        checklist.append(("app.py imports successfully", True, None))
    except Exception as e:
        checklist.append(("app.py imports successfully", False, str(e)))
        print(f"FAILED to import app: {e}")
        return checklist

    # 2. Test model loads successfully
    try:
        assert app.MODEL is not None, "Model is None"
        assert hasattr(app.MODEL, "predict_proba"), "Model missing predict_proba method"
        checklist.append(("model loads successfully", True, f"Type: {type(app.MODEL).__name__}"))
    except Exception as e:
        checklist.append(("model loads successfully", False, str(e)))

    # 3. Test WordNet loads successfully
    try:
        assert len(app.VOCAB) >= 40000, f"Unexpected vocab size: {len(app.VOCAB)}"
        checklist.append(("WordNet loads successfully", True, f"Vocabulary: {len(app.VOCAB):,} words"))
    except Exception as e:
        checklist.append(("WordNet loads successfully", False, str(e)))

    # 4. Test game starts
    try:
        state = app.init_game("Medium (7–9 letters)")
        assert 7 <= len(state["secret"]) <= 9, f"Secret length out of range: {len(state['secret'])}"
        assert len(state["pattern"]) == len(state["secret"])
        assert all(ch == "_" for ch in state["pattern"])
        assert state["wrong_count"] == 0
        assert not state["game_over"]
        checklist.append(("game starts", True, f"Secret length: {len(state['secret'])}"))
    except Exception as e:
        checklist.append(("game starts", False, str(e)))

    # 5. Test New Game works
    try:
        state1 = app.init_game()
        state2 = app.init_game()
        assert "secret" in state1 and "secret" in state2
        assert state2["wrong_count"] == 0
        assert state2["pattern"] == ["_"] * len(state2["secret"])
        checklist.append(("New Game works", True, None))
    except Exception as e:
        checklist.append(("New Game works", False, str(e)))

    # 6. Test valid letter works (correct guess)
    try:
        test_state = {
            "secret": "planet",
            "pattern": ["_", "_", "_", "_", "_", "_"],
            "guessed": set(),
            "wrong_count": 0,
            "game_over": False,
            "won": False,
            "message": "",
        }
        updated_state, msg = app.apply_guess(test_state, "P")
        assert "p" in updated_state["guessed"], "Letter 'p' not in guessed"
        assert updated_state["pattern"][0] == "p", "Pattern index 0 not revealed"
        assert updated_state["wrong_count"] == 0, "Wrong count should remain 0"
        checklist.append(("valid letter works", True, f"Pattern: {' '.join(updated_state['pattern'])}"))
    except Exception as e:
        checklist.append(("valid letter works", False, str(e)))

    # 7. Test incorrect letter works
    try:
        test_state = {
            "secret": "planet",
            "pattern": ["p", "_", "_", "_", "_", "_"],
            "guessed": {"p"},
            "wrong_count": 0,
            "game_over": False,
            "won": False,
            "message": "",
        }
        updated_state, msg = app.apply_guess(test_state, "Z")
        assert "z" in updated_state["guessed"], "Letter 'z' not in guessed"
        assert updated_state["wrong_count"] == 1, f"Wrong count expected 1, got {updated_state['wrong_count']}"
        checklist.append(("incorrect letter works", True, f"Wrong count: {updated_state['wrong_count']}"))
    except Exception as e:
        checklist.append(("incorrect letter works", False, str(e)))

    # 8. Test repeated letter is handled
    try:
        test_state = {
            "secret": "planet",
            "pattern": ["p", "_", "_", "_", "_", "_"],
            "guessed": {"p"},
            "wrong_count": 0,
            "game_over": False,
            "won": False,
            "message": "",
        }
        updated_state, msg = app.apply_guess(test_state, "P")
        assert "already guessed" in msg.lower(), f"Unexpected message: {msg}"
        assert updated_state["wrong_count"] == 0, "Repeated guess should not penalize attempts"
        checklist.append(("repeated letter is handled", True, msg))
    except Exception as e:
        checklist.append(("repeated letter is handled", False, str(e)))

    # 9. Test invalid input is handled
    try:
        test_state = {
            "secret": "planet",
            "pattern": ["_", "_", "_", "_", "_", "_"],
            "guessed": set(),
            "wrong_count": 0,
            "game_over": False,
            "won": False,
            "message": "",
        }
        _, msg1 = app.apply_guess(test_state, "123")
        _, msg2 = app.apply_guess(test_state, "")
        _, msg3 = app.apply_guess(test_state, "ab")
        assert "single letter" in msg1.lower()
        assert "single letter" in msg2.lower()
        assert "single letter" in msg3.lower()
        checklist.append(("invalid input is handled", True, "Rejected non-letters, blanks, and multi-chars"))
    except Exception as e:
        checklist.append(("invalid input is handled", False, str(e)))

    # 10. Test AI prediction works
    try:
        ranked, top_letter, conf = app.rank_letters(
            word_length=6,
            pattern=["_", "_", "A", "_", "_", "_"],
            guessed_letters={"a", "e", "r"},
            top_k=5
        )
        assert len(ranked) > 0, "Ranked list is empty"
        assert top_letter is not None, "Top letter is None"
        assert conf > 0.0, f"Confidence score should be > 0, got {conf}"
        checklist.append(("AI prediction works", True, f"Top pick: {top_letter.upper()} ({conf:.1f}%)"))
    except Exception as e:
        checklist.append(("AI prediction works", False, str(e)))

    # 11. Test already-guessed and already-revealed letters are excluded
    try:
        guessed = {"a", "e", "i", "o", "u", "s", "t"}
        pattern = ["s", "t", "_", "_", "_", "_"]
        ranked, top_letter, _ = app.rank_letters(
            word_length=6,
            pattern=pattern,
            guessed_letters=guessed,
            top_k=26
        )
        predicted_letters = set(c for c, _, _ in ranked)
        forbidden = guessed | set(c for c in pattern if c != "_")
        overlap = predicted_letters & forbidden
        assert len(overlap) == 0, f"Predictions contained already guessed/revealed letters: {overlap}"
        checklist.append(("already-guessed letters are excluded", True, "Zero overlap with guessed/revealed"))
    except Exception as e:
        checklist.append(("already-guessed letters are excluded", False, str(e)))

    # 12. Test Top-5 predictions work
    try:
        ranked, _, _ = app.rank_letters(
            word_length=7,
            pattern=["_", "_", "_", "_", "_", "_", "_"],
            guessed_letters=set(),
            top_k=5
        )
        assert len(ranked) == 5, f"Expected 5 predictions, got {len(ranked)}"
        # Verify sorted descending by confidence
        scores = [conf for _, conf, _ in ranked]
        assert scores == sorted(scores, reverse=True), f"Not sorted descending: {scores}"
        formatted = ", ".join(f"{c.upper()}:{conf:.1f}%" for c, conf, _ in ranked)
        checklist.append(("Top-5 predictions work", True, formatted))
    except Exception as e:
        checklist.append(("Top-5 predictions work", False, str(e)))

    # 13. Test WIN condition works
    try:
        test_state = {
            "secret": "cat",
            "pattern": ["c", "a", "_"],
            "guessed": {"c", "a"},
            "wrong_count": 1,
            "game_over": False,
            "won": False,
            "message": "",
        }
        updated_state, msg = app.apply_guess(test_state, "t")
        assert updated_state["game_over"] is True, "Game should be game_over"
        assert updated_state["won"] is True, "Game should be won"
        assert "win" in updated_state["message"].lower() or "win" in msg.lower()
        checklist.append(("WIN condition works", True, updated_state["message"]))
    except Exception as e:
        checklist.append(("WIN condition works", False, str(e)))

    # 14. Test GAME OVER condition works
    try:
        test_state = {
            "secret": "cat",
            "pattern": ["c", "_", "_"],
            "guessed": {"c", "b", "d", "e", "f", "g"},
            "wrong_count": 5,
            "game_over": False,
            "won": False,
            "message": "",
        }
        updated_state, msg = app.apply_guess(test_state, "z")
        assert updated_state["wrong_count"] == 6, f"Expected wrong_count 6, got {updated_state['wrong_count']}"
        assert updated_state["game_over"] is True, "Game should be game_over"
        assert updated_state["won"] is False, "Game should not be won"
        assert "game over" in updated_state["message"].lower()
        checklist.append(("GAME OVER condition works", True, updated_state["message"]))
    except Exception as e:
        checklist.append(("GAME OVER condition works", False, str(e)))

    # 15. Test requirements.txt is complete
    try:
        with open("requirements.txt", "r") as f:
            reqs = f.read()
        for pkg in ["gradio", "nltk", "scikit-learn", "pandas", "numpy"]:
            assert pkg in reqs, f"Missing {pkg} in requirements.txt"
        checklist.append(("requirements.txt is complete", True, "All required packages present"))
    except Exception as e:
        checklist.append(("requirements.txt is complete", False, str(e)))

    # 16. Test GitHub and Hugging Face README are complete
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            readme = f.read()
        assert "sdk: gradio" in readme, "Missing HF Space sdk in README.md"
        assert "app_file: app.py" in readme, "Missing app_file in README.md"
        assert "# WordNet AI Hangman" in readme, "Missing title in README.md"
        assert "## 🚀 Live Demo" in readme, "Missing Live Demo section in README.md"
        assert "YOUR_HUGGING_FACE_SPACE_URL" in readme, "Missing placeholder in README.md"
        for section in ["Project Overview", "Features", "How the AI Works", "WordNet Corpus",
                        "ML Model Architecture", "Training", "Game Workflow", "Installation",
                        "Running Locally", "Hugging Face Deployment", "Project Structure",
                        "Technologies Used", "Results", "Future Improvements", "Author"]:
            assert section in readme, f"Missing section '{section}' in README.md"
        checklist.append(("GitHub & Hugging Face README is complete", True, "All 16 sections + YAML frontmatter verified"))
    except Exception as e:
        checklist.append(("GitHub & Hugging Face README is complete", False, str(e)))

    # Print summary
    print("\nVerification Checklist Results:")
    all_passed = True
    for item, passed, details in checklist:
        status_symbol = "[PASS]" if passed else "[FAIL]"
        status_text = "OK" if passed else "FAILED"
        detail_text = f" ({details})" if details else ""
        print(f"{status_symbol} {item}: {status_text}{detail_text}")
        if not passed:
            all_passed = False

    print("\nOverall Status:", "ALL TESTS PASSED! SUCCESS" if all_passed else "SOME TESTS FAILED!")
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
