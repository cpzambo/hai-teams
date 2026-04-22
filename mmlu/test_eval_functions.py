"""
Unit tests for extract_final_answer and score_response functions
used in both xai_eval.py and llama_eval.py (mmlu).

Run with:
    python test_eval_functions.py
or:
    python -m pytest test_eval_functions.py -v
"""

import unittest
import re
import sys
import os

# ── copy the two functions here so tests are self-contained ──────────────────

def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    result = match.group(1).strip() if match else model_output.strip()
    return result.strip("\"'`").strip()

def score_response(model_response, gold_letter):
    final_answer = extract_final_answer(model_response)
    if final_answer is None:
        return 0
    # case 1: exact match "A" == "A"
    if final_answer.upper().strip() == gold_letter.upper().strip():
        return 1
    # case 2: model says "A." or "(A)" or "A)" — must be only a letter, no full words
    m = re.match(r'^\(?([A-D])\)?\.?\s*$', final_answer.strip())
    if m and m.group(1).upper() == gold_letter.upper():
        return 1
    return 0

# ── tests ────────────────────────────────────────────────────────────────────

class TestExtractFinalAnswer(unittest.TestCase):

    def test_normal(self):
        """Standard model output with reasoning before the answer."""
        out = "Some reasoning here.\nFinal Answer: B"
        self.assertEqual(extract_final_answer(out), "B")

    def test_case_insensitive(self):
        """'final answer:' in lowercase should still be matched."""
        out = "I think the answer is C.\nfinal answer: C"
        self.assertEqual(extract_final_answer(out), "C")

    def test_strips_double_quotes(self):
        """Model wraps the letter in double quotes."""
        out = 'Final Answer: "A"'
        self.assertEqual(extract_final_answer(out), "A")

    def test_strips_single_quotes(self):
        out = "Final Answer: 'D'"
        self.assertEqual(extract_final_answer(out), "D")

    def test_strips_backticks(self):
        out = "Final Answer: `C`"
        self.assertEqual(extract_final_answer(out), "C")

    def test_no_match_returns_full_output(self):
        """If no 'Final Answer:' tag, return the whole stripped output."""
        out = "   The answer is B   "
        self.assertEqual(extract_final_answer(out), "The answer is B")

    def test_extra_whitespace(self):
        """Extra spaces between colon and letter are stripped."""
        out = "Final Answer:    A"
        self.assertEqual(extract_final_answer(out), "A")


class TestScoreResponse(unittest.TestCase):

    # ── should score 1 ───────────────────────────────────────────────────────

    def test_exact_match(self):
        self.assertEqual(score_response("Final Answer: A", "A"), 1)

    def test_lowercase_answer(self):
        """Model returns lowercase letter, gold is uppercase."""
        self.assertEqual(score_response("Final Answer: b", "B"), 1)

    def test_parentheses_format(self):
        """Model returns '(C)' instead of 'C'."""
        self.assertEqual(score_response("Final Answer: (C)", "C"), 1)

    def test_letter_with_period(self):
        """Model returns 'D.' instead of 'D'."""
        self.assertEqual(score_response("Final Answer: D.", "D"), 1)

    def test_letter_with_closing_paren(self):
        """Model returns 'B)' instead of 'B'."""
        self.assertEqual(score_response("Final Answer: B)", "B"), 1)

    def test_reasoning_before_answer(self):
        """Model gives reasoning then the correct letter."""
        resp = "The subject is ethics so the answer must be C.\nFinal Answer: C"
        self.assertEqual(score_response(resp, "C"), 1)

    def test_quoted_letter(self):
        """Model wraps letter in quotes: Final Answer: \"A\"."""
        self.assertEqual(score_response('Final Answer: "A"', "A"), 1)

    # ── should score 0 ───────────────────────────────────────────────────────

    def test_wrong_letter(self):
        self.assertEqual(score_response("Final Answer: B", "A"), 0)

    def test_full_choice_text_not_matched(self):
        """Model returns the full choice text instead of the letter — scored 0
        because MMLU eval expects a letter (A-D) as the gold."""
        self.assertEqual(score_response("Final Answer: Cryptocurrencies", "C"), 0)

    def test_empty_response(self):
        self.assertEqual(score_response("", "A"), 0)


# ── run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)