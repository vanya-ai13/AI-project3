"""
vectorizer.py — Vocabulary normalisation + TF-IDF vectorization.

Why TF-IDF over raw binary (bag-of-words) vectors?
----------------------------------------------------
Binary vectors treat every matching skill equally — a role that lists 10 skills
gets the same weight on "Python" as one that lists 2.  TF-IDF down-weights terms
that appear in *many* roles (ubiquitous skills like "Python" or "SQL" are less
discriminative) and up-weights terms that appear in *few* roles (niche skills
like "Apache Spark" or "Penetration Testing" are more specific).  The result is
a richer distance metric that better separates specialist roles.

Why cosine similarity (handled in scoring.py) over Euclidean distance?
-----------------------------------------------------------------------
TF-IDF document vectors are of variable length — a job role with 8 skill tags
produces a denser vector than one with 3.  Euclidean distance penalises vector
magnitude, so a verbose role description appears "further" from the user profile
even when they share the same set of concepts.  Cosine similarity is magnitude-
invariant: it measures the angle between vectors, capturing directional similarity
independent of how many terms the description contains.

Synonym map rationale:
-----------------------
Natural language skill tags fragment the vocabulary space.  "Web Design" and
"Frontend Development" are semantically the same career concept but produce
disjoint TF-IDF dimensions.  We collapse near-synonyms to a canonical form
*before* vectorization so the model sees one dimension, not two half-filled ones.
The map is intentionally small and curated — it is NOT a full ontology.  Adding
unmeasured expansions would pollute the vocabulary space.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Synonym / near-synonym normalisation map.
# Keys are lowercase raw tokens; values are the canonical replacement term.
# Add entries here whenever a data audit reveals vocabulary fragmentation.
# ---------------------------------------------------------------------------
SYNONYM_MAP: dict[str, str] = {
    "web design": "frontend development",
    "front end": "frontend development",
    "frontend": "frontend development",
    "back end": "backend development",
    "backend": "backend development",
    "ml": "machine learning",
    "ai": "machine learning",
    "devops": "devops engineering",
    "sre": "site reliability engineering",
    "k8s": "kubernetes",
    "infra as code": "infrastructure as code",
    "iac": "infrastructure as code",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "aws": "aws",
    "gcp": "gcp",
    "azure": "azure",
    "postgres": "postgresql",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "react.js": "react",
    "reactjs": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "automation testing": "test automation",
    "qa": "quality assurance",
    "rest api": "rest apis",
    "rest": "rest apis",
    "api": "rest apis",
    "nosql": "mongodb",
    "android development": "android",
    "ios development": "ios",
}


def normalise_skill(skill: str) -> str:
    """
    Lowercase, strip whitespace, and resolve near-synonyms for a single skill token.

    The synonym map is applied on the lowercased, stripped form.  If no entry
    is found the cleaned token is returned as-is.  This ensures every skill that
    enters the vectorizer has been through the same normalisation path, regardless
    of whether it came from the user or the dataset.

    Args:
        skill: A raw skill string (e.g. "Web Design", "  ML  ").

    Returns:
        Canonical, lowercased, stripped skill string.
    """
    cleaned = skill.lower().strip()
    return SYNONYM_MAP.get(cleaned, cleaned)


def normalise_skill_list(skills: list[str]) -> list[str]:
    """
    Apply normalise_skill to an entire list of skill tokens.

    Args:
        skills: List of raw skill strings.

    Returns:
        List of normalised, deduplicated skill strings (order preserved for
        determinism; duplicates removed after normalisation).
    """
    seen: set[str] = set()
    result: list[str] = []
    for s in skills:
        normed = normalise_skill(s)
        if normed not in seen:
            seen.add(normed)
            result.append(normed)
    return result


def skills_to_document(skills: list[str]) -> str:
    """
    Join a list of normalised skill tokens into a single space-separated string.

    TfidfVectorizer expects string documents, not token lists.  Joining on a
    space (rather than a comma) avoids the analyser treating punctuation as a
    token boundary, which matters for multi-word skills like "machine learning".

    Assumption: skills have already been normalised (lowercased, synonym-resolved).

    Args:
        skills: Normalised skill list for a single entity (user or job role).

    Returns:
        Space-separated string document ready for TfidfVectorizer.
    """
    return " ".join(skills)


class VectorizerResult(NamedTuple):
    """Container for the fitted vectorizer and resulting TF-IDF matrices."""

    vectorizer: TfidfVectorizer
    role_matrix: np.ndarray   # shape (n_roles, n_features)
    user_vector: np.ndarray   # shape (1, n_features)
    role_names: list[str]     # parallel to role_matrix rows
    is_user_zero_vector: bool # True → user cold-start detected


def build_tfidf_vectors(
    user_skills: list[str],
    dataset: list[dict],
) -> VectorizerResult:
    """
    Fit a single TfidfVectorizer on the combined corpus (job roles + user),
    then return the resulting matrices.

    Why a shared fit?
    -----------------
    Fitting separate vectorizers for the user and the dataset would produce
    different vocabulary indices — the same term "python" at column 5 in the
    role matrix might land at column 12 in the user vector.  Cosine similarity
    would then compare incompatible dimensions.  We fit *once* on the joint
    corpus so every entity shares the identical vocabulary space.

    Cold start detection:
    ---------------------
    If the user's skill tokens do not overlap with any term in the fitted
    vocabulary (e.g. the user entered extremely niche or misspelled skills),
    the user TF-IDF vector will be all zeros.  We detect this here and set
    `is_user_zero_vector=True` so the pipeline can route to a fallback.

    Args:
        user_skills: Raw (not yet normalised) skill strings from the user.
                     Normalisation is applied inside this function.
        dataset:     List of job-role dicts from ingestion.load_dataset().

    Returns:
        VectorizerResult named tuple.
    """
    # --- 1. Normalise all skill lists ---
    normed_user_skills = normalise_skill_list(user_skills)
    normalised_dataset: list[dict] = [
        {"job_role": rec["job_role"], "skills": normalise_skill_list(rec["skills"])}
        for rec in dataset
    ]

    # --- 2. Build string documents for each entity ---
    user_doc = skills_to_document(normed_user_skills)
    role_docs = [skills_to_document(rec["skills"]) for rec in normalised_dataset]
    role_names = [rec["job_role"] for rec in normalised_dataset]

    # Joint corpus: role documents first, then the user document.
    # The vectorizer is fit on the joint set so all entities share a vocabulary.
    joint_corpus = role_docs + [user_doc]

    # --- 3. Fit TfidfVectorizer on the joint corpus ---
    # token_pattern: default sklearn pattern skips single-character tokens.
    # We use a custom pattern to preserve short acronyms like "ml", "qa", "js".
    # sublinear_tf=True applies log(1+tf) term frequency scaling, which reduces
    # the disproportionate weight of repeated terms in longer descriptions.
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w[\w./+-]*\b",
        sublinear_tf=True,
        lowercase=True,   # belt-and-suspenders; we already lowercased in normalise
    )
    tfidf_matrix = vectorizer.fit_transform(joint_corpus)

    # Split back into role matrix and user vector.
    n_roles = len(role_docs)
    role_matrix: np.ndarray = tfidf_matrix[:n_roles].toarray()
    user_vector: np.ndarray = tfidf_matrix[n_roles:].toarray()  # shape (1, n_features)

    # Cold-start check: all-zero user vector means no vocabulary overlap.
    is_user_zero_vector = bool(np.allclose(user_vector, 0.0))

    return VectorizerResult(
        vectorizer=vectorizer,
        role_matrix=role_matrix,
        user_vector=user_vector,
        role_names=role_names,
        is_user_zero_vector=is_user_zero_vector,
    )
