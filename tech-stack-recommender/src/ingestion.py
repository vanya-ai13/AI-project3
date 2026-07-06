"""
ingestion.py — Step 1 of the 4-step recommendation pipeline.

Responsibilities:
  - Validate raw user skill input (must be >= 3 skills).
  - Load and parse the raw_skills.csv dataset into a structured format.

Design note: Keeping ingestion separate from vectorization ensures each pipeline
stage is independently testable and replaceable (e.g., swap CSV for a database
source without touching the vectorizer).
"""

from __future__ import annotations

import csv
from pathlib import Path


class InsufficientSkillsError(ValueError):
    """Raised when the user provides fewer than the minimum required skills."""


def validate_user_skills(user_skills: list[str], min_skills: int = 3) -> list[str]:
    """
    Validate and lightly normalise the raw user skill list.

    Normalisation applied here (lowercasing and stripping) is the minimum
    needed so that callers don't need to think about case sensitivity.
    The vocabulary-level synonym resolution happens later in vectorizer.py.

    Args:
        user_skills: Raw skill strings provided by the user.
        min_skills:  Minimum acceptable number of skills (default 3).

    Returns:
        Cleaned list of skill strings.

    Raises:
        TypeError: If user_skills is not a list.
        InsufficientSkillsError: If fewer than min_skills non-empty skills are given.
    """
    if not isinstance(user_skills, list):
        raise TypeError(
            f"user_skills must be a list, got {type(user_skills).__name__!r}."
        )

    cleaned = [s.strip() for s in user_skills if isinstance(s, str) and s.strip()]

    if len(cleaned) < min_skills:
        raise InsufficientSkillsError(
            f"At least {min_skills} skills are required; "
            f"got {len(cleaned)} non-empty skill(s): {cleaned!r}. "
            "Add more skills to receive meaningful recommendations."
        )

    return cleaned


def load_dataset(dataset_path: str | Path) -> list[dict]:
    """
    Load and parse the job-role dataset from a CSV file.

    Expected CSV format:
        job_role,skills
        Data Scientist,Python,Machine Learning,Statistics,...

    The 'skills' field may span multiple CSV columns (one skill per column)
    OR be a single comma-separated string — this parser handles both layouts
    by collecting every column after 'job_role' as individual skill tokens.

    Args:
        dataset_path: Path to the raw_skills.csv file.

    Returns:
        List of dicts with keys:
            - "job_role" (str): Name of the job role.
            - "skills"   (list[str]): Cleaned skill tags for that role.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: If the CSV is missing the required 'job_role' column.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {path.resolve()}. "
            "Ensure raw_skills.csv is present in the data/ directory."
        )

    records: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None or not header:
            raise ValueError("Dataset CSV appears to be empty.")

        # Normalise the header to detect the job_role column position.
        normalised_header = [col.strip().lower() for col in header]
        if "job_role" not in normalised_header:
            raise ValueError(
                "Dataset CSV must have a 'job_role' column as the first column."
            )
        role_idx = normalised_header.index("job_role")

        for row in reader:
            if not row or not row[role_idx].strip():
                continue  # skip blank rows

            job_role = row[role_idx].strip()

            # All columns after job_role are treated as individual skill tokens.
            raw_skill_tokens = row[role_idx + 1 :]
            skills = [tok.strip() for tok in raw_skill_tokens if tok.strip()]

            records.append({"job_role": job_role, "skills": skills})

    if not records:
        raise ValueError(
            f"Dataset at {path.resolve()} loaded zero job roles. "
            "Verify the CSV has data rows below the header."
        )

    return records
