"""
Unit tests for extract_final_answer and score_response in xai_eval.py.

Run with:
    python test_xai_eval.py
or:
    python -m pytest test_xai_eval.py -v
"""

import unittest
import re

# ── copy functions from xai_eval.py so tests are self-contained ──────────────

def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    result = match.group(1).strip() if match else model_output.strip()
    return result.strip("\"'`").strip()

def score_response(model_response, gold_text, gold_letter, answer_index):
    final_answer = extract_final_answer(model_response)
    if not final_answer:
        return 0

    fa = final_answer.strip()

    # case 1: exact match with full choice text (same as openai)
    if fa.lower() == gold_text.lower():
        return 1

    # case 2: letter only — "A", "(A)", "A.", "A)" — must be only a letter
    m = re.match(r'^\(?([A-D])\)?\.?\s*$', fa, re.IGNORECASE)
    if m and m.group(1).upper() == gold_letter.upper():
        return 1

    # case 3: index number — model says "2" instead of "C"
    if fa.strip() == str(answer_index):
        return 1

    # case 4: letter + choice text — "A. Cryptocurrencies, Cheap..."
    m = re.match(r'^\(?([A-D])\)?[.):]?\s+(.+)', fa, re.IGNORECASE | re.DOTALL)
    if m and m.group(1).upper() == gold_letter.upper():
        return 1

    # case 5: comma/space variation of full text — "barn, damp" vs "barn damp"
    if fa.lower().replace(",", " ").split() == gold_text.lower().replace(",", " ").split():
        return 1

    return 0

# ── shared fixture: answer index=2 → letter "C" ───────────────────────────────
GOLD_TEXT   = "Cryptocurrencies, Cheap, Secure, Financial crime"
GOLD_LETTER = "C"
ANSWER_IDX  = 2

def score(resp):
    return score_response(resp, GOLD_TEXT, GOLD_LETTER, ANSWER_IDX)

# ── tests: extract_final_answer ───────────────────────────────────────────────

class TestExtractFinalAnswer(unittest.TestCase):

    def test_normal(self):
        self.assertEqual(extract_final_answer("Some reasoning.\nFinal Answer: B"), "B")

    def test_case_insensitive(self):
        self.assertEqual(extract_final_answer("final answer: C"), "C")

    def test_strips_double_quotes(self):
        self.assertEqual(extract_final_answer('Final Answer: "A"'), "A")

    def test_strips_single_quotes(self):
        self.assertEqual(extract_final_answer("Final Answer: 'D'"), "D")

    def test_strips_backticks(self):
        self.assertEqual(extract_final_answer("Final Answer: `C`"), "C")

    def test_no_match_returns_full_output(self):
        self.assertEqual(extract_final_answer("   The answer is B   "), "The answer is B")

    def test_extra_whitespace(self):
        self.assertEqual(extract_final_answer("Final Answer:    A"), "A")

# ── tests: score_response ─────────────────────────────────────────────────────

class TestScoreResponse(unittest.TestCase):

    # should score 1 ──────────────────────────────────────────────────────────

    def test_letter_exact(self):
        self.assertEqual(score("Final Answer: C"), 1)

    def test_letter_lowercase(self):
        self.assertEqual(score("Final Answer: c"), 1)

    def test_letter_parentheses(self):
        self.assertEqual(score("Final Answer: (C)"), 1)

    def test_letter_with_period(self):
        self.assertEqual(score("Final Answer: C."), 1)

    def test_letter_with_closing_paren(self):
        self.assertEqual(score("Final Answer: C)"), 1)

    def test_index_number(self):
        """Model returns the index number '2' instead of letter 'C'."""
        self.assertEqual(score("Final Answer: 2"), 1)

    def test_full_choice_text(self):
        """Model returns the full choice text (same as openai gold)."""
        self.assertEqual(score(f"Final Answer: {GOLD_TEXT}"), 1)

    def test_full_text_case_insensitive(self):
        self.assertEqual(score("Final Answer: cryptocurrencies, cheap, secure, financial crime"), 1)

    def test_letter_plus_text(self):
        """Model returns 'C. Cryptocurrencies...'."""
        self.assertEqual(score(f"Final Answer: C. {GOLD_TEXT}"), 1)

    def test_letter_paren_plus_text(self):
        """Model returns '(C) Cryptocurrencies...'."""
        self.assertEqual(score(f"Final Answer: (C) {GOLD_TEXT}"), 1)

    def test_reasoning_before_answer(self):
        self.assertEqual(score("Based on my analysis.\nFinal Answer: C"), 1)

    def test_quoted_letter(self):
        self.assertEqual(score('Final Answer: "C"'), 1)

    # should score 0 ──────────────────────────────────────────────────────────

    def test_wrong_letter(self):
        self.assertEqual(score("Final Answer: A"), 0)

    def test_wrong_index(self):
        self.assertEqual(score("Final Answer: 0"), 0)

    def test_wrong_full_text(self):
        self.assertEqual(score("Final Answer: Cryptocurrencies, Expensive, Secure, Financial Crime"), 0)

    def test_word_starting_with_correct_letter(self):
        """'Cryptocurrencies' starts with C but must not be treated as letter C."""
        self.assertEqual(score("Final Answer: Cryptocurrencies"), 0)

    def test_empty_response(self):
        self.assertEqual(score(""), 0)

# ── run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
