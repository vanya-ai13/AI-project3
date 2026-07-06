"""
recommender.py — Public API: orchestrates the 4-step pipeline.

Pipeline:
  Step 1 — Ingestion   (ingestion.validate_user_skills + ingestion.load_dataset)
  Step 2 — Scoring     (vectorizer.build_tfidf_vectors + scoring.score_roles)
  Step 3 — Sorting     (ranking.sort_scores)
  Step 4 — Filtering   (ranking.top_n)
  Fallback             (cold_start.cold_start_fallback — triggered on zero vector)

This module is the only import consumers need.  All internal modules are an
implementation detail.
"""

from __future__ import annotations

from ingestion import load_dataset, validate_user_skills
from vectorizer import build_tfidf_vectors
from scoring import score_roles
from ranking import sort_scores, top_n
from cold_start import cold_start_fallback


def recommend(
    user_skills: list[str],
    top_n_count: int = 3,
    dataset_path: str = "data/raw_skills.csv",
) -> list[dict]:
    """
    Content-based filtering recommendation engine.

    Maps a user's raw skill/interest tags to a ranked list of job-role
    recommendations using TF-IDF vectorization + cosine similarity.

    Pipeline (4 discrete, independently testable steps):
      1. Ingestion   — validate user input; load dataset from CSV.
      2. Scoring     — build shared TF-IDF vocabulary; compute cosine similarity
                       for every job role against the user vector.
      3. Sorting     — sort descending by similarity score.
      4. Filtering   — truncate to top_n_count results.

    Cold start handling:
      If the user's skills produce an all-zero TF-IDF vector (no vocabulary
      overlap), the function falls back to a "most tag-rich roles" default
      rather than crashing or returning an empty list.  Fallback scores are
      set to 0.0 so callers can distinguish them from genuine cosine scores.

    Args:
        user_skills:   Raw skill/interest strings from the user.
                       Must contain at least 3 non-empty strings.
        top_n_count:   Maximum number of recommendations to return (default 3).
        dataset_path:  Path to the raw_skills.csv dataset file.

    Returns:
        List of dicts, length <= top_n_count, sorted descending by score:
            [{"job_role": "DevOps Engineer", "score": 0.87}, ...]

    Raises:
        InsufficientSkillsError (subclass of ValueError):
            If len(user_skills) < 3.
        FileNotFoundError:
            If dataset_path does not point to an existing file.
        ValueError:
            If the dataset CSV is malformed or contains no job roles.
    """
    # --- Step 1: Ingestion ---
    cleaned_skills = validate_user_skills(user_skills)
    dataset = load_dataset(dataset_path)

    # --- Step 2: Vectorize + Score ---
    vec_result = build_tfidf_vectors(cleaned_skills, dataset)

    if vec_result.is_user_zero_vector:
        # Cold start: user skills don't appear in the dataset vocabulary.
        # Return a ranked default rather than an empty or errored response.
        return cold_start_fallback(dataset, top_n=top_n_count)

    scored = score_roles(vec_result.user_vector, vec_result.role_matrix, vec_result.role_names)

    # --- Step 3: Sort ---
    sorted_roles = sort_scores(scored)

    # --- Step 4: Filter ---
    return top_n(sorted_roles, n=top_n_count)
