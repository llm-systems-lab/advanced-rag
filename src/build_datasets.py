"""Generate the HyDE dataset under ``experiments/datasets``.

``hyde.json`` stores only the generated ``hypothetical_answer`` per question,
keyed by the question ``id`` (its index in ``data/evaluation/questions.json``).
The base questions are never duplicated.

``step_back.json`` holds the authored step-back rewrites in the same id-keyed
shape and is not regenerated here.

Usage:
    uv run python src/build_datasets.py [--force]
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from dataset_utils import DATASETS_DIR, load_questions, write_json
from ingestion import PROJECT_ROOT
from techniques.query_transformation.hyde import generate_hypothetical_answer


def build_hyde(questions: list[dict]) -> list[dict]:
    """Generate a hypothetical answer per question with the Groq model."""
    records: list[dict] = []
    for item in questions:
        print(f"  generating HyDE passage: {item['user_input'][:60]}")
        records.append(
            {
                "id": item["id"],
                "hypothetical_answer": generate_hypothetical_answer(item["user_input"]),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the HyDE dataset")
    parser.add_argument(
        "--force", action="store_true", help="Regenerate the HyDE passages"
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    hyde_path = DATASETS_DIR / "hyde.json"
    if hyde_path.exists() and not args.force:
        print(f"Keeping existing {hyde_path} (use --force to regenerate)")
        return

    write_json(hyde_path, build_hyde(load_questions()))


if __name__ == "__main__":
    main()
