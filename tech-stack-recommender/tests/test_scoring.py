"""
test_scoring.py — Unit tests for src/scoring.py.

Coverage targets:
  - score_roles: correct shape, float scores in [0, 1], input validation.
  - Semantic correctness: a highly-overlapping user profile MUST rank above
    a loosely-overlapping one — this is the core quality gate.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scoring import score_roles
from vectorizer import build_tfidf_vectors


# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

DATASET = [
    {"job_role": "Data Scientist",     "skills": ["Python", "Machine Learning", "Statistics", "Data Analysis", "Pandas"]},
    {"job_role": "DevOps Engineer",    "skills": ["Docker", "Kubernetes", "CI/CD", "Linux", "Terraform", "Ansible"]},
    {"job_role": "Frontend Developer", "skills": ["JavaScript", "React", "CSS", "HTML", "TypeScript"]},
]


def _vectorize(user_skills: list[str]) -> tuple:
    """Helper: build vectors and return (user_vector, role_matrix, role_names)."""
    result = build_tfidf_vectors(user_skills, DATASET)
    return result.user_vector, result.role_matrix, result.role_names


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestScoreRolesOutputStructure:
    def test_returns_list_of_dicts(self):
        uv, rm, rn = _vectorize(["Python", "Machine Learning", "Statistics"])
        scores = score_roles(uv, rm, rn)
        assert isinstance(scores, list)
        assert all(isinstance(s, dict) for s in scores)

    def test_one_entry_per_role(self):
        uv, rm, rn = _vectorize(["Python", "Machine Learning", "Statistics"])
        scores = score_roles(uv, rm, rn)
        assert len(scores) == len(DATASET)

    def test_dict_keys_are_job_role_and_score(self):
        uv, rm, rn = _vectorize(["Python", "Machine Learning", "Statistics"])
        scores = score_roles(uv, rm, rn)
        for entry in scores:
            assert set(entry.keys()) == {"job_role", "score"}

    def test_scores_are_floats(self):
        uv, rm, rn = _vectorize(["Python", "Machine Learning", "Statistics"])
        scores = score_roles(uv, rm, rn)
        for entry in scores:
            assert isinstance(entry["score"], float)

    def test_scores_in_valid_range(self):
        """Cosine similarity must be in [0, 1] for non-negative TF-IDF vectors."""
        uv, rm, rn = _vectorize(["Python", "Machine Learning", "Statistics"])
        scores = score_roles(uv, rm, rn)
        for entry in scores:
            assert 0.0 <= entry["score"] <= 1.0, (
                f"Score out of range for {entry['job_role']}: {entry['score']}"
            )

    def test_role_names_preserved(self):
        uv, rm, rn = _vectorize(["Python", "Machine Learning", "Statistics"])
        scores = score_roles(uv, rm, rn)
        returned_names = {s["job_role"] for s in scores}
        assert returned_names == {r["job_role"] for r in DATASET}


# ---------------------------------------------------------------------------
# SEMANTIC CORRECTNESS — the critical quality gate
# ---------------------------------------------------------------------------

class TestSemanticRanking:
    def test_high_overlap_scores_above_low_overlap(self):
        """
        A user with Python/Machine Learning/Statistics skills must score HIGHER
        against 'Data Scientist' than against 'DevOps Engineer'.

        This test is the core quality gate for the recommendation engine.
        If this fails, the vectorizer or similarity metric is broken — not just
        cosmetically wrong but fundamentally wrong for the use case.
        """
        uv, rm, rn = _vectorize(["Python", "Machine Learning", "Statistics"])
        scores = score_roles(uv, rm, rn)

        score_map = {s["job_role"]: s["score"] for s in scores}
        assert score_map["Data Scientist"] > score_map["DevOps Engineer"], (
            f"Expected Data Scientist ({score_map['Data Scientist']:.4f}) "
            f"> DevOps Engineer ({score_map['DevOps Engineer']:.4f}). "
            "Semantic ranking is broken."
        )

    def test_frontend_user_scores_above_data_scientist(self):
        """
        A user with JavaScript/React/CSS skills must score HIGHER against
        'Frontend Developer' than against 'Data Scientist'.
        """
        uv, rm, rn = _vectorize(["JavaScript", "React", "CSS"])
        scores = score_roles(uv, rm, rn)

        score_map = {s["job_role"]: s["score"] for s in scores}
        assert score_map["Frontend Developer"] > score_map["Data Scientist"], (
            f"Expected Frontend Developer ({score_map['Frontend Developer']:.4f}) "
            f"> Data Scientist ({score_map['Data Scientist']:.4f})."
        )

    def test_devops_user_scores_above_frontend_developer(self):
        """
        A user with Docker/Kubernetes/Linux skills must score HIGHER against
        'DevOps Engineer' than against 'Frontend Developer'.
        """
        uv, rm, rn = _vectorize(["Docker", "Kubernetes", "Linux"])
        scores = score_roles(uv, rm, rn)

        score_map = {s["job_role"]: s["score"] for s in scores}
        assert score_map["DevOps Engineer"] > score_map["Frontend Developer"], (
            f"Expected DevOps Engineer ({score_map['DevOps Engineer']:.4f}) "
            f"> Frontend Developer ({score_map['Frontend Developer']:.4f})."
        )

    def test_perfect_match_scores_higher_than_partial_match(self):
        """
        A user who exactly mirrors a role's skills should score higher than a user
        with only one overlapping skill.

        Design note: Both user vectors MUST be produced from the SAME fitted
        TfidfVectorizer so they share an identical feature space.  Calling
        build_tfidf_vectors twice produces two independent fits with different
        vocabulary sizes — the resulting vectors are incompatible, and score_roles
        correctly raises a ValueError.  We therefore fit once on the full-match
        user, then use vectorizer.transform() to project the partial-match user
        into that same feature space.
        """
        from vectorizer import normalise_skill_list, skills_to_document

        # Fit on the full-match user (largest vocabulary — ensures partial-match
        # skills that happen to be in-vocabulary are still represented).
        full_result = build_tfidf_vectors(
            ["Python", "Machine Learning", "Statistics", "Data Analysis", "Pandas"],
            DATASET,
        )

        # Project partial-match user through the SAME fitted vectorizer.
        partial_doc = skills_to_document(
            normalise_skill_list(["Python", "Cloud Computing", "Automation"])
        )
        partial_match_uv = full_result.vectorizer.transform([partial_doc]).toarray()

        full_scores    = score_roles(full_result.user_vector, full_result.role_matrix, full_result.role_names)
        partial_scores = score_roles(partial_match_uv,        full_result.role_matrix, full_result.role_names)

        full_ds    = next(s["score"] for s in full_scores    if s["job_role"] == "Data Scientist")
        partial_ds = next(s["score"] for s in partial_scores if s["job_role"] == "Data Scientist")

        assert full_ds > partial_ds, (
            f"Full match ({full_ds:.4f}) should outscore partial match ({partial_ds:.4f}) "
            "against Data Scientist."
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestScoreRolesValidation:
    def test_mismatched_role_names_raises(self):
        """Passing a role_names list that doesn't match role_matrix rows must raise."""
        uv, rm, rn = _vectorize(["Python", "Machine Learning", "Statistics"])
        with pytest.raises(ValueError, match="parallel"):
            score_roles(uv, rm, rn[:-1])  # one fewer name than rows

    def test_feature_dim_mismatch_raises(self):
        """Passing vectors with incompatible feature dims must raise."""
        uv, rm, rn = _vectorize(["Python", "Machine Learning", "Statistics"])
        bad_matrix = rm[:, :-5]  # truncate features to force mismatch
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            score_roles(uv, bad_matrix, rn)
