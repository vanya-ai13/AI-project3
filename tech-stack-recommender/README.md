# Tech Stack Recommender

A self-contained **content-based filtering engine** that maps a user's raw skills and interests to ranked career/job-role recommendations using **TF-IDF vectorization** and **cosine similarity**. No collaborative filtering, no LLM calls, no external API dependencies — deterministic, explainable, and unit-tested.

---

## Architecture — Input → Process → Output

```
INPUT (User State)          PROCESS (Similarity Logic)         OUTPUT (Top-N List)
────────────────────        ───────────────────────────        ────────────────────
- 3+ raw skill strings   →  - Build shared vocabulary       →  - Sorted job-role list
- from raw_skills.csv       - TF-IDF weight items + user        - Truncated to Top N
                            - Cosine similarity per item        - Similarity score shown
                            - Sort descending
```

The pipeline is implemented as **4 discrete, independently testable steps**:

| Step | Module | Responsibility |
|------|--------|---------------|
| 1 | `src/ingestion.py` | Validate user input (≥3 skills); load & parse CSV dataset |
| 2 | `src/vectorizer.py` + `src/scoring.py` | TF-IDF vectorize joint corpus; compute cosine similarity |
| 3 | `src/ranking.py` | Sort descending by similarity score |
| 4 | `src/ranking.py` | Truncate to Top-N |
| ⚡ | `src/cold_start.py` | Fallback when user vector is all-zero |

---

## Why TF-IDF Over Raw Binary Vectors?

Binary (bag-of-words) vectors treat every matching skill equally. A role listing 10 skills gets the same weight on "Python" as one with 2. **TF-IDF** addresses this by:

- **Down-weighting** ubiquitous terms ("Python", "SQL") that appear in many roles and are therefore less discriminative.
- **Up-weighting** niche terms ("Apache Spark", "Penetration Testing") that appear in few roles and are highly specific.

Result: a richer distance metric that better separates specialist roles from generalist ones.

---

## Why Cosine Similarity Over Euclidean Distance?

TF-IDF vectors have **variable magnitude**: a role with 8 skill tags produces a denser, higher-magnitude vector than one with 3. Euclidean distance is magnitude-sensitive — it would judge a verbose role description as "further" from the user profile even when they share identical concepts.

**Cosine similarity** measures the angle θ between vectors:

```
cosine(u, v) = (u · v) / (‖u‖ × ‖v‖)
```

The magnitude terms cancel out, leaving a score in **[0, 1]** that captures directional agreement — "do these vectors point the same way in skill space?" — independent of description length. This is the correct metric for variable-length skill profiles.

---

## Cold Start Handling

| Scenario | Trigger | Fallback |
|----------|---------|----------|
| **User cold start** | User's skills produce all-zero TF-IDF vector (no vocabulary overlap) | Return roles ranked by tag count (broadest skill profile = most generalist) with `score=0.0` |
| **Item cold start** | New job role with sparse/no tags | Scoreable as soon as it has ≥1 recognised tag — no historical interaction data needed |

---

## Project Structure

```
tech-stack-recommender/
├── data/
│   └── raw_skills.csv         # 15 job roles, 5-8 skill tags each
├── src/
│   ├── ingestion.py            # Step 1: validate + load
│   ├── vectorizer.py           # Step 2a: TF-IDF vectorization + synonym map
│   ├── scoring.py              # Step 2b: cosine similarity scoring
│   ├── ranking.py              # Steps 3+4: sort + top-N filter
│   ├── cold_start.py           # Fallback logic
│   └── recommender.py          # Public API: orchestrates the pipeline
├── tests/
│   ├── test_vectorizer.py
│   ├── test_scoring.py
│   ├── test_ranking.py
│   └── test_cold_start.py
├── cli.py                      # CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `scikit-learn>=1.4.0`, `numpy>=1.26.0`, `pytest>=8.0.0`

---

## Usage

### CLI

Run from the project root (`tech-stack-recommender/`):

```bash
# Basic usage — 3 skills, default Top-3
python cli.py --skills Python "Cloud Computing" Automation

# Specify more results
python cli.py --skills Python Docker Kubernetes --top-n 5

# Custom dataset path
python cli.py --skills JavaScript React CSS --dataset data/raw_skills.csv
```

**Example output:**
```
┌─────────────────────────────────────────────────┐
│        Tech Stack Recommender — Results         │
└─────────────────────────────────────────────────┘
  Input skills : Python, Cloud Computing, Automation
  Top 3 match(es):

  1. DevOps Engineer               0.6214  [████████████░░░░░░░░]
  2. Site Reliability Engineer     0.5831  [███████████░░░░░░░░░]
  3. Cloud Architect               0.5102  [██████████░░░░░░░░░░]
```

### Python API

```python
import sys
sys.path.insert(0, "src")
from recommender import recommend

results = recommend(
    user_skills=["Python", "Cloud Computing", "Automation"],
    top_n_count=3,
    dataset_path="data/raw_skills.csv",
)
# [{"job_role": "DevOps Engineer", "score": 0.87}, ...]
```

---

## Running Tests

```bash
# From the project root
python -m pytest tests/ -v
```

The test suite includes:
- **Semantic correctness** assertions — a Python/ML/Stats user must rank "Data Scientist" above "DevOps Engineer".
- **Zero-vector cold-start** integration test — verifies the fallback path fires and returns `score=0.0` entries.
- **Input validation** — confirms typed exceptions with actionable messages for malformed input.

---

## Dataset

`data/raw_skills.csv` contains 15 job roles with 5–8 realistic skill tags each:

| Job Role | Sample Skills |
|----------|--------------|
| Data Scientist | Python, Machine Learning, Statistics, Pandas |
| DevOps Engineer | Docker, Kubernetes, CI/CD, Terraform |
| ML Engineer | TensorFlow, PyTorch, MLOps, Data Pipelines |
| Frontend Developer | JavaScript, React, TypeScript, CSS |
| Site Reliability Engineer | Linux, Monitoring, Kubernetes, Automation |
| … and 10 more | — |

---

## Design Decisions & Assumptions

- **Shared vocabulary fit**: The TF-IDF vectorizer is fit on the joint corpus (all job-role documents + the user document) in a single pass. Fitting separately would produce different vocabulary indices, making cosine similarity across the two matrices meaningless.
- **Synonym map**: A curated `SYNONYM_MAP` in `vectorizer.py` collapses near-synonyms ("ML" → "machine learning", "k8s" → "kubernetes") before vectorization. This prevents vocabulary fragmentation where the same concept occupies two disjoint dimensions.
- **`sublinear_tf=True`**: Log-scaled term frequency reduces the disproportionate weight of repeated terms in longer role descriptions.
- **Cold-start scores = 0.0**: Fallback results use `score=0.0` as a sentinel value so downstream callers can programmatically distinguish fallback results from genuine cosine similarity scores.
