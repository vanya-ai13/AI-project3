"""
scoring.py — Step 2 of the 4-step recommendation pipeline.

Computes cosine similarity between the user TF-IDF vector and every
job-role TF-IDF vector produced by vectorizer.build_tfidf_vectors().

Why cosine similarity over Euclidean distance?
-----------------------------------------------
In a TF-IDF vector space, documents (job roles) differ not only in *which*
terms they contain but also in *how many* terms they list.  A role with 8
skill tags produces a denser, higher-magnitude vector than a role with 3 tags.
Euclidean distance is sensitive to this magnitude difference — it would judge
a verbose role description as "further" from the user profile even when they
share the exact same concepts.

Cosine similarity measures the angle θ between two vectors:

    cosine(u, v) = (u · v) / (||u|| * ||v||)

The magnitude terms (||u||, ||v||) cancel out, leaving a score in [0, 1] that
captures *directional* agreement — i.e. "do these vectors point in the same
direction in the skill space?" — independent of how many terms each description
contains.  This makes it the correct metric for variable-length skill profiles.

sklearn.metrics.pairwise.cosine_similarity is used (not hand-rolled) because:
  - It is numerically stable (handles zero vectors without divide-by-zero).
  - It operates on full arrays in one BLAS call → efficient for large datasets.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def score_roles(
    user_vector: np.ndarray,
    role_matrix: np.ndarray,
    role_names: list[str],
) -> list[dict]:
    """
    Compute cosine similarity between the user vector and every role vector.

    Args:
        user_vector: Shape (1, n_features) — the TF-IDF user profile vector.
        role_matrix: Shape (n_roles, n_features) — TF-IDF matrix for all roles.
        role_names:  List of job-role name strings, parallel to role_matrix rows.

    Returns:
        List of dicts, one per job role:
            [{"job_role": "DevOps Engineer", "score": 0.87}, ...]
        Scores are Python floats rounded to 4 decimal places for readability.
        The list is NOT yet sorted (sorting is the responsibility of ranking.py).

    Raises:
        ValueError: If role_matrix row count does not match len(role_names).
        ValueError: If user_vector and role_matrix have incompatible feature dims.
    """
    if role_matrix.shape[0] != len(role_names):
        raise ValueError(
            f"role_matrix has {role_matrix.shape[0]} rows but "
            f"role_names has {len(role_names)} entries. They must be parallel."
        )

    if user_vector.shape[1] != role_matrix.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: user_vector has {user_vector.shape[1]} "
            f"features but role_matrix has {role_matrix.shape[1]}. "
            "Both must come from the same fitted TfidfVectorizer."
        )

    # cosine_similarity returns shape (1, n_roles); flatten to a 1-D array.
    similarity_scores: np.ndarray = cosine_similarity(user_vector, role_matrix).flatten()

    return [
        {"job_role": name, "score": round(float(score), 4)}
        for name, score in zip(role_names, similarity_scores)
    ]
