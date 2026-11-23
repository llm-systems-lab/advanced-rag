"""Dataset helpers for the base question set and per-technique artifacts.

``data/evaluation/questions.json`` holds ``id``, ``user_input`` and ``reference``.
Technique-specific generated fields live under ``experiments/datasets`` where
each record is ``{"id": <question id>, <generated_field>: ...}``, aligned with
the base question set by ``id`` so the original questions are never duplicated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ingestion import PROJECT_ROOT

QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
DATASETS_DIR = PROJECT_ROOT / "experiments" / "datasets"


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {len(records)} records to {path}")


def load_questions() -> list[dict[str, Any]]:
    """Load the base labelled questions (``id``, ``user_input``, ``reference``)."""
    return read_json(QUESTIONS_PATH)


def load_artifacts(path: Path) -> dict[int, dict[str, Any]]:
    """Load a technique dataset as ``{question_id: record}``."""
    return {int(record["id"]): record for record in read_json(path)}


def load_dataset(artifacts_path: Path | None = None) -> list[dict[str, Any]]:
    """Merge a technique dataset's generated fields into the base questions.

    The ``id`` field itself is not merged. When ``artifacts_path`` is missing or
    omitted, the bare base questions are returned.
    """
    questions = load_questions()
    if artifacts_path is None or not artifacts_path.exists():
        return questions

    artifacts = load_artifacts(artifacts_path)
    merged: list[dict[str, Any]] = []
    for question in questions:
        extra = {
            key: value
            for key, value in artifacts.get(int(question["id"]), {}).items()
            if key != "id"
        }
        merged.append({**question, **extra})
    return merged
