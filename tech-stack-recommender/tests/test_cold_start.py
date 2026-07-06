"""
test_cold_start.py — Unit tests for src/cold_start.py.

Coverage targets:
  - cold_start_fallback: correct ordering by tag count, score=0.0 sentinel,
    top_n truncation, empty dataset error, zero-vector integration test.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cold_start import cold_start_fallback


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATASET = [
    {"job_role": "Generalist",   "skills": ["Python", "SQL", "Docker", "Linux", "AWS", "Git"]},
    {"job_role": "Specialist",   "skills": ["TensorFlow", "PyTorch"]},
    {"job_role": "Mid-Level",    "skills": ["React", "Node.js", "PostgreSQL", "CSS"]},
]


# ---------------------------------------------------------------------------
# cold_start_fallback
# ---------------------------------------------------------------------------

class TestColdStartFallback:
    def test_returns_list_of_dicts(self):
        result = cold_start_fallback(DATASET, top_n=3)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_dict_keys_are_job_role_and_score(self):
        result = cold_start_fallback(DATASET, top_n=3)
        for entry in result:
            assert set(entry.keys()) == {"job_role", "score"}

    def test_all_scores_are_zero(self):
        """Fallback scores must be 0.0 to distinguish them from real cosine scores."""
        result = cold_start_fallback(DATASET, top_n=3)
        for entry in result:
            assert entry["score"] == 0.0, (
                f"Expected score=0.0 for cold-start fallback entry, "
                f"got {entry['score']} for {entry['job_role']!r}."
            )

    def test_ordered_by_tag_count_descending(self):
        """
        The role with the most skill tags should appear first.
        'Generalist' has 6 tags > 'Mid-Level' (4) > 'Specialist' (2).
        """
        result = cold_start_fallback(DATASET, top_n=3)
        names = [r["job_role"] for r in result]
        assert names[0] == "Generalist",  f"Expected Generalist first, got {names[0]!r}"
        assert names[1] == "Mid-Level",   f"Expected Mid-Level second, got {names[1]!r}"
        assert names[2] == "Specialist",  f"Expected Specialist third, got {names[2]!r}"

    def test_truncates_to_top_n(self):
        result = cold_start_fallback(DATASET, top_n=2)
        assert len(result) == 2

    def test_top_n_larger_than_dataset_returns_all(self):
        result = cold_start_fallback(DATASET, top_n=100)
        assert len(result) == len(DATASET)

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="empty dataset"):
            cold_start_fallback([], top_n=3)

    def test_default_top_n_is_three(self):
        result = cold_start_fallback(DATASET)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# Integration: zero-vector triggers cold start in the full pipeline
# ---------------------------------------------------------------------------

class TestZeroVectorColdStartIntegration:
    def test_recommender_falls_back_on_zero_vector(self, tmp_path):
        """
        End-to-end: when all user skills are tokeniser-invisible (single chars),
        the vectorizer should detect a zero vector and recommender should route
        to the fallback rather than crashing.

        We construct the zero-vector condition by passing skills that consist
        only of single characters, which sklearn's token_pattern (\\b\\w[\\w.+-]*\\b)
        drops, resulting in an empty token list and an all-zero TF-IDF row.
        """
        import csv

        # Build a minimal dataset CSV in a temp directory.
        csv_path = tmp_path / "raw_skills.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["job_role", "skills"])
            writer.writerow(["Data Scientist", "Python", "Machine Learning", "Statistics"])
            writer.writerow(["DevOps Engineer", "Docker", "Kubernetes", "Linux"])

        # Import here to avoid circular import during collection.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from recommender import recommend

        # Single-character "skills" are stripped by the tokeniser → zero vector.
        result = recommend(
            user_skills=["a", "b", "c"],
            top_n_count=2,
            dataset_path=str(csv_path),
        )

        assert isinstance(result, list)
        assert len(result) <= 2
        # Fallback entries always have score=0.0.
        assert all(r["score"] == 0.0 for r in result), (
            "Expected cold-start fallback scores of 0.0; "
            f"got {[r['score'] for r in result]}"
        )
