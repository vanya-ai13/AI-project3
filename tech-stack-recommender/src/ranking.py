"""
ranking.py — Steps 3 & 4 of the 4-step recommendation pipeline.

Step 3: Sort the scored role list descending by similarity score.
Step 4: Truncate to the Top-N result set.

These are kept as two discrete functions (not merged) so that callers can
sort without truncating (e.g. for full-rank logging) or verify the sort
independently in tests.
"""

from __future__ import annotations


def sort_scores(scored_roles: list[dict]) -> list[dict]:
    """
    Sort a list of scored role dicts descending by similarity score.

    Args:
        scored_roles: List of dicts with keys 'job_role' (str) and 'score' (float),
                      as returned by scoring.score_roles().

    Returns:
        New list sorted descending by 'score'.  The input list is NOT mutated.

    Raises:
        KeyError: If any dict in scored_roles is missing the 'score' key.
    """
    return sorted(scored_roles, key=lambda r: r["score"], reverse=True)


def top_n(sorted_roles: list[dict], n: int = 3) -> list[dict]:
    """
    Truncate a sorted role list to the Top-N entries.

    Truncation is the final step before returning results to the caller.
    Keeping it separate from sorting makes both steps unit-testable in isolation.

    Args:
        sorted_roles: Role list already sorted descending by score
                      (output of sort_scores()).
        n:            Maximum number of results to return.  Must be >= 1.

    Returns:
        Slice of sorted_roles with length <= n.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError(
            f"top_n requires n >= 1; got n={n!r}. "
            "Use n=1 for a single best-match recommendation."
        )
    return sorted_roles[:n]
