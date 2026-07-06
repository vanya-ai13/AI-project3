"""
test_ranking.py — Unit tests for src/ranking.py.

Coverage targets:
  - sort_scores: descending order, stability, empty input.
  - top_n: correct truncation, boundary cases, invalid n.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ranking import sort_scores, top_n


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

UNSORTED = [
    {"job_role": "Frontend Developer", "score": 0.3},
    {"job_role": "Data Scientist",     "score": 0.9},
    {"job_role": "DevOps Engineer",    "score": 0.6},
    {"job_role": "ML Engineer",        "score": 0.75},
]


# ---------------------------------------------------------------------------
# sort_scores
# ---------------------------------------------------------------------------

class TestSortScores:
    def test_sorts_descending(self):
        result = sort_scores(UNSORTED)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_entry_is_highest_score(self):
        result = sort_scores(UNSORTED)
        assert result[0]["job_role"] == "Data Scientist"
        assert result[0]["score"] == 0.9

    def test_bottom_entry_is_lowest_score(self):
        result = sort_scores(UNSORTED)
        assert result[-1]["job_role"] == "Frontend Developer"
        assert result[-1]["score"] == 0.3

    def test_does_not_mutate_input(self):
        original_order = [r["job_role"] for r in UNSORTED]
        sort_scores(UNSORTED)
        assert [r["job_role"] for r in UNSORTED] == original_order

    def test_empty_list_returns_empty(self):
        assert sort_scores([]) == []

    def test_single_element_list(self):
        single = [{"job_role": "Data Scientist", "score": 0.9}]
        assert sort_scores(single) == single

    def test_equal_scores_preserved(self):
        """Ties should not raise; order among ties is unspecified but stable."""
        tied = [
            {"job_role": "A", "score": 0.5},
            {"job_role": "B", "score": 0.5},
        ]
        result = sort_scores(tied)
        assert len(result) == 2
        assert all(r["score"] == 0.5 for r in result)


# ---------------------------------------------------------------------------
# top_n
# ---------------------------------------------------------------------------

class TestTopN:
    def test_truncates_to_n(self):
        sorted_roles = sort_scores(UNSORTED)
        result = top_n(sorted_roles, n=3)
        assert len(result) == 3

    def test_returns_highest_scores(self):
        sorted_roles = sort_scores(UNSORTED)
        result = top_n(sorted_roles, n=2)
        assert result[0]["score"] == 0.9
        assert result[1]["score"] == 0.75

    def test_n_equals_list_length_returns_all(self):
        sorted_roles = sort_scores(UNSORTED)
        result = top_n(sorted_roles, n=len(UNSORTED))
        assert len(result) == len(UNSORTED)

    def test_n_larger_than_list_returns_all(self):
        """top_n should not crash if n > len(sorted_roles); just return all."""
        sorted_roles = sort_scores(UNSORTED)
        result = top_n(sorted_roles, n=100)
        assert len(result) == len(UNSORTED)

    def test_n_equals_one(self):
        sorted_roles = sort_scores(UNSORTED)
        result = top_n(sorted_roles, n=1)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_invalid_n_zero_raises(self):
        with pytest.raises(ValueError, match="n >= 1"):
            top_n(sort_scores(UNSORTED), n=0)

    def test_invalid_n_negative_raises(self):
        with pytest.raises(ValueError, match="n >= 1"):
            top_n(sort_scores(UNSORTED), n=-5)

    def test_default_n_is_three(self):
        """Default top_n should return at most 3 items."""
        sorted_roles = sort_scores(UNSORTED)
        result = top_n(sorted_roles)
        assert len(result) <= 3
