"""
cli.py — Command-line interface for the Tech Stack Recommender.

Usage:
    python cli.py --skills Python "Cloud Computing" Automation
    python cli.py --skills Python Docker Kubernetes --top-n 5
    python cli.py --skills Python Docker Kubernetes --dataset data/raw_skills.csv

Run from the project root (tech-stack-recommender/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to path so the CLI works when invoked from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ingestion import InsufficientSkillsError
from recommender import recommend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "Tech Stack Recommender — content-based filtering engine.\n"
            "Maps your skills to ranked job-role recommendations using "
            "TF-IDF vectorization + cosine similarity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py --skills Python Docker Kubernetes\n"
            '  python cli.py --skills Python "Cloud Computing" Automation --top-n 5\n'
            "  python cli.py --skills JavaScript React CSS --dataset data/raw_skills.csv"
        ),
    )
    parser.add_argument(
        "--skills",
        nargs="+",
        required=True,
        metavar="SKILL",
        help="Space-separated list of your skills/interests (minimum 3). "
             "Wrap multi-word skills in quotes, e.g. \"Cloud Computing\".",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        metavar="N",
        dest="top_n",
        help="Number of top recommendations to display (default: 3).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw_skills.csv",
        metavar="PATH",
        help="Path to the raw_skills.csv dataset file (default: data/raw_skills.csv).",
    )
    return parser


def format_results(results: list[dict], user_skills: list[str]) -> str:
    """Format the recommendation results as a human-readable ASCII string."""
    lines: list[str] = []
    lines.append("")
    lines.append("+---------------------------------------------------+")
    lines.append("|        Tech Stack Recommender -- Results          |")
    lines.append("+---------------------------------------------------+")
    lines.append(f"  Input skills : {', '.join(user_skills)}")
    lines.append(f"  Top {len(results)} match(es):")
    lines.append("")

    is_fallback = all(r["score"] == 0.0 for r in results)
    if is_fallback:
        lines.append(
            "  [!] Cold start detected: your skills didn't match the dataset vocabulary.\n"
            "      Showing most popular (tag-rich) roles as a default:"
        )
        lines.append("")

    for rank, entry in enumerate(results, start=1):
        role  = entry["job_role"]
        score = entry["score"]
        if is_fallback:
            bar = "-" * 20
            score_str = "N/A (cold start)"
        else:
            # Visual bar: 20 '=' chars ~ score 1.0
            filled = round(score * 20)
            bar = "=" * filled + "-" * (20 - filled)
            score_str = f"{score:.4f}"
        lines.append(f"  {rank}. {role:<32} {score_str}  [{bar}]")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        results = recommend(
            user_skills=args.skills,
            top_n_count=args.top_n,
            dataset_path=args.dataset,
        )
    except InsufficientSkillsError as exc:
        parser.error(str(exc))
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))

    print(format_results(results, args.skills))


if __name__ == "__main__":
    main()
