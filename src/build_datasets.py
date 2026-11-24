"""Generate the LLM-derived technique datasets under ``experiments/datasets``.

Each dataset stores only the generated fields per question, keyed by the
question ``id``; the base questions are never duplicated.

- ``hyde.json``: ``hypothetical_answer`` (see HyDE).
- ``multi_query.json``: ``queries`` (three paraphrases for RAG Fusion).

``step_back.json`` holds authored step-back rewrites and is not regenerated here.

Usage:
    uv run python src/build_datasets.py [--only hyde|multi_query] [--force]
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from dataset_utils import (
    DATASETS_DIR,
    Question,
    QuestionArtifact,
    load_questions,
    write_json,
)
from ingestion import PROJECT_ROOT
from techniques.query_transformation.hyde import generate_hypothetical_answer
from techniques.query_transformation.multi_query import generate_multiple_queries
from tracing import flush


def build_hyde(questions: list[Question]) -> list[QuestionArtifact]:
    """Generate a hypothetical answer per question with the Groq model."""
    records: list[QuestionArtifact] = []
    for item in questions:
        print(f"  [hyde] {item['user_input'][:60]}")
        records.append(
            {
                "id": item["id"],
                "hypothetical_answer": generate_hypothetical_answer(item["user_input"]),
            }
        )
    return records


def build_multi_query(questions: list[Question]) -> list[QuestionArtifact]:
    """Generate paraphrased queries per question with the Groq model."""
    records: list[QuestionArtifact] = []
    for item in questions:
        print(f"  [multi_query] {item['user_input'][:60]}")
        records.append(
            {
                "id": item["id"],
                "queries": generate_multiple_queries(item["user_input"]),
            }
        )
    return records


BUILDERS = {"hyde": build_hyde, "multi_query": build_multi_query}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate technique datasets")
    parser.add_argument(
        "--only", choices=sorted(BUILDERS), help="Build a single dataset"
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing datasets"
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    questions = load_questions()
    targets = [args.only] if args.only else list(BUILDERS)

    for name in targets:
        path = DATASETS_DIR / f"{name}.json"
        if path.exists() and not args.force:
            print(f"Keeping existing {path} (use --force to regenerate)")
            continue
        write_json(path, BUILDERS[name](questions))

    flush()


if __name__ == "__main__":
    main()
