"""Dataset helpers for the base question set and per-technique artifacts.

``data/evaluation/questions.json`` holds ``id``, ``user_input`` and ``reference``.
Technique-specific generated fields live under ``experiments/datasets`` where
each record is ``{"id": <question id>, <generated_field>: ...}``, aligned with
the base question set by ``id`` so the original questions are never duplicated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from ingestion import PROJECT_ROOT

QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
DATASETS_DIR = PROJECT_ROOT / "experiments" / "datasets"


class Question(TypedDict):
    """A labelled evaluation question, optionally enriched by a technique."""

    id: int
    user_input: str
    reference: str
    step_back_question: NotRequired[str]
    hypothetical_answer: NotRequired[str]
    queries: NotRequired[list[str]]


class QuestionArtifact(TypedDict):
    """Technique-generated fields for one question, keyed by question ``id``."""

    id: int
    step_back_question: NotRequired[str]
    hypothetical_answer: NotRequired[str]
    queries: NotRequired[list[str]]


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return cast(object, json.load(handle))


def write_json(path: Path, records: list[QuestionArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {len(records)} records to {path}")


def load_questions() -> list[Question]:
    """Load the base labelled questions (``id``, ``user_input``, ``reference``)."""
    return cast(list[Question], read_json(QUESTIONS_PATH))


def load_artifacts(path: Path) -> dict[int, QuestionArtifact]:
    """Load a technique dataset as ``{question_id: record}``."""
    records = cast(list[QuestionArtifact], read_json(path))
    return {int(record["id"]): record for record in records}


def load_dataset(artifacts_path: Path | None = None) -> list[Question]:
    """Merge a technique dataset's generated fields into the base questions.

    The ``id`` field itself is not merged. When ``artifacts_path`` is missing or
    omitted, the bare base questions are returned.
    """
    questions = load_questions()
    if artifacts_path is None or not artifacts_path.exists():
        return questions

    artifacts = load_artifacts(artifacts_path)
    merged: list[Question] = []
    for question in questions:
        record = artifacts.get(int(question["id"]))
        if record is None:
            merged.append(question)
            continue
        extra = {key: value for key, value in record.items() if key != "id"}
        merged.append(cast(Question, {**question, **extra}))
    return merged
