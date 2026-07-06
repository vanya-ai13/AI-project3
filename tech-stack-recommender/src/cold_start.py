"""
cold_start.py — Fallback logic for the user cold-start problem.

Cold start problem (content-based variant):
--------------------------------------------
In collaborative filtering, cold start typically means "no interaction history."
In content-based filtering the analogous problem occurs when the user's skill
profile produces an all-zero TF-IDF vector — i.e. none of their listed skills
appear anywhere in the shared vocabulary built from the job-role dataset.

This can happen when:
  - The user enters highly niche / misspelled skills not present in the dataset.
  - The user's skills exist but are filtered out by the tokeniser (e.g. all
    single-character tokens, which sklearn's default token_pattern drops).

Why NOT return an error or an empty list?
------------------------------------------
From a UX standpoint, returning nothing is worse than returning a ranked default.
The user came for a recommendation; a graceful fallback ("here are popular roles")
is always more useful than silence or a crash.

Fallback strategy — "most tag-rich" roles:
-------------------------------------------
Without a user signal, the safest heuristic is to surface roles that have the
most skill tags (broadest appeal / most descriptive profiles).  This acts as a
proxy for "trending / generalist" roles and avoids picking an arbitrary default
that might not be relevant to any real user.

We deliberately do NOT use hardcoded role names as fallbacks — that would be
brittle and dataset-dependent.  Instead we count tags from the live dataset so
the fallback stays correct even after the CSV is updated.
"""

from __future__ import annotations


def cold_start_fallback(dataset: list[dict], top_n: int = 3) -> list[dict]:
    """
    Return a ranked list of job roles ordered by number of skill tags (descending).

    This fallback is triggered when the user's TF-IDF vector is all-zero,
    meaning their input skills do not overlap with the dataset vocabulary.

    The 'score' field is set to 0.0 for all fallback entries to make it
    unambiguous to downstream callers (and to users reading CLI output) that
    these are *defaults*, not cosine similarity scores.

    Args:
        dataset: List of job-role dicts from ingestion.load_dataset().
        top_n:   Maximum number of fallback roles to return.

    Returns:
        List of dicts [{"job_role": str, "score": 0.0}, ...] sorted descending
        by tag count, truncated to top_n.

    Raises:
        ValueError: If dataset is empty.
    """
    if not dataset:
        raise ValueError(
            "Cannot compute cold-start fallback from an empty dataset. "
            "Ensure raw_skills.csv has at least one job role."
        )

    ranked = sorted(dataset, key=lambda r: len(r.get("skills", [])), reverse=True)
    return [{"job_role": r["job_role"], "score": 0.0} for r in ranked[:top_n]]
