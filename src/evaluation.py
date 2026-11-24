"""RAGAS evaluation of the RAG pipeline.

Runs the retrieval + generation pipeline over a labelled question set and scores
the answers with RAGAS metrics, using a Groq (default) or OpenRouter judge model.

Usage:
    uv run python src/evaluation.py [experiment_name]
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ragas import (
    EvaluationDataset,
    MultiTurnSample,
    RunConfig,
    SingleTurnSample,
    evaluate,
)
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from dataset_utils import DATASETS_DIR, Question, load_dataset
from generation import Answer, answer
from ingestion import PROJECT_ROOT
from tracing import flush

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


class EvaluationPayload(TypedDict):
    """Serialised RAGAS results for one experiment."""

    experiment: str
    num_questions: int
    scores: dict[str, float | None]
    per_question: list[dict[str, object]]


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Copy .env.example to .env and fill it in."
        )
    return value


def build_judge() -> LangchainLLMWrapper:
    """Return the judge model: Groq by default, OpenRouter when selected."""
    load_dotenv(PROJECT_ROOT / ".env")
    llm: BaseChatModel
    if os.getenv("JUDGE_PROVIDER", "groq").lower() == "openrouter":
        llm = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL") or _env("GROQ_MODEL"),
            api_key=SecretStr(_env("OPENROUTER_API_KEY")),
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
        )
    else:
        llm = ChatGroq(
            model=_env("GROQ_MODEL"),
            api_key=SecretStr(_env("GROQ_API_KEY")),
            temperature=0,
        )
    return LangchainLLMWrapper(llm)


def build_embeddings() -> LangchainEmbeddingsWrapper:
    """Return Ollama embeddings used by the answer-relevancy metric."""
    load_dotenv(PROJECT_ROOT / ".env")
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=_env("OLLAMA_EMBEDDING_MODEL"))
    )


def _standard_answer(item: Question) -> Answer:
    return answer(item["user_input"])


def build_dataset(
    questions: list[Question], answer_fn: Callable[[Question], Answer]
) -> EvaluationDataset:
    """Run the pipeline for every question and collect RAGAS samples."""
    samples: list[SingleTurnSample | MultiTurnSample] = []
    for item in questions:
        result = answer_fn(item)
        samples.append(
            SingleTurnSample(
                user_input=item["user_input"],
                retrieved_contexts=[doc.page_content for doc in result.sources],
                response=result.text,
                reference=item["reference"],
            )
        )
        print(f"  answered: {item['user_input'][:70]}")
    return EvaluationDataset(samples=samples)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    to_item = getattr(value, "item", None)
    if callable(to_item):
        value = to_item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_experiment(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(
            f"Invalid experiment name '{name}'. Use only letters, digits, '_' and '-'."
        )
    return name


def run_evaluation(
    experiment: str = "baseline",
    answer_fn: Callable[[Question], Answer] = _standard_answer,
    artifacts_path: Path | None = None,
) -> EvaluationPayload:
    """Evaluate the pipeline and persist the results under experiments/results."""
    experiment = _safe_experiment(experiment)
    questions = load_dataset(artifacts_path)
    print(f"Evaluating {len(questions)} questions with experiment '{experiment}'")

    answer_relevancy.strictness = 1
    dataset = build_dataset(questions, answer_fn)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=build_judge(),
        embeddings=build_embeddings(),
        run_config=RunConfig(timeout=300, max_retries=5, max_wait=120, max_workers=2),
        show_progress=False,
    )
    assert isinstance(result, EvaluationResult)

    frame = result.to_pandas()
    scores: dict[str, float | None] = {
        name: float(frame[name].mean()) if name in frame.columns else None
        for name in METRIC_NAMES
    }
    payload: EvaluationPayload = {
        "experiment": experiment,
        "num_questions": len(questions),
        "scores": cast(dict[str, float | None], _json_safe(scores)),
        "per_question": cast(
            list[dict[str, object]], _json_safe(frame.to_dict(orient="records"))
        ),
    }

    experiment_dir = RESULTS_DIR / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    results_path = experiment_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved results to {results_path}")

    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument("experiment", nargs="?", help="Experiment name")
    parser.add_argument(
        "--technique",
        choices=[
            "standard",
            "step_back",
            "hyde",
            "multi_query",
            "hybrid",
            "rerank",
            "compression",
        ],
        default="standard",
        help="RAG technique to evaluate",
    )
    args = parser.parse_args()

    if args.technique == "step_back":
        from techniques.query_transformation.step_back import answer_step_back

        answer_fn: Callable[[Question], Answer] = lambda item: answer_step_back(
            item["user_input"], item.get("step_back_question")
        )
        default_experiment = "step_back"
        artifacts_path = DATASETS_DIR / "step_back.json"
    elif args.technique == "hyde":
        from techniques.query_transformation.hyde import answer_hyde

        answer_fn = lambda item: answer_hyde(
            item["user_input"], item.get("hypothetical_answer")
        )
        default_experiment = "hyde"
        artifacts_path = DATASETS_DIR / "hyde.json"
    elif args.technique == "multi_query":
        from techniques.query_transformation.multi_query import answer_multi_query

        answer_fn = lambda item: answer_multi_query(
            item["user_input"], item.get("queries")
        )
        default_experiment = "multi_query"
        artifacts_path = DATASETS_DIR / "multi_query.json"
    elif args.technique == "hybrid":
        from techniques.retrieval.hybrid import answer_hybrid

        answer_fn = lambda item: answer_hybrid(item["user_input"])
        default_experiment = "hybrid"
        artifacts_path = None
    elif args.technique == "rerank":
        from techniques.retrieval.rerank import answer_rerank

        answer_fn = lambda item: answer_rerank(item["user_input"])
        default_experiment = "rerank"
        artifacts_path = None
    elif args.technique == "compression":
        from techniques.context_optimization.compression import answer_compress

        answer_fn = lambda item: answer_compress(item["user_input"])
        default_experiment = "compression"
        artifacts_path = None
    else:
        answer_fn = _standard_answer
        default_experiment = "baseline"
        artifacts_path = None

    if artifacts_path is not None and not artifacts_path.exists():
        print(f"Dataset {artifacts_path} not found; running without it")
        artifacts_path = None

    payload = run_evaluation(
        args.experiment or default_experiment,
        answer_fn=answer_fn,
        artifacts_path=artifacts_path,
    )
    print("\nScores:")
    for name, value in payload["scores"].items():
        shown = f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
        print(f"  {name}: {shown}")
    flush()


if __name__ == "__main__":
    main()
