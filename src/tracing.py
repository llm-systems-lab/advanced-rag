"""Langfuse tracing for the RAG pipeline.

Tracing is configured entirely through environment variables
(``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, ``LANGFUSE_BASE_URL``). When
the keys are absent the Langfuse client disables itself and these helpers become
cheap no-ops, so the pipeline keeps working without tracing.

Langfuse is imported lazily, after ``.env`` is loaded, so the client never
initializes with missing credentials.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from langchain_core.documents import Document

from ingestion import load_env

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

TRACE_NAME = "rag-answer"
RETRIEVAL_NAME = "retrieve-context"


class Observation(Protocol):
    """Minimal surface shared by the Langfuse observations used here."""

    def update(self, *, output: object = ...) -> object: ...


def tracing_enabled() -> bool:
    """Return whether Langfuse credentials are configured."""
    load_env()
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    """Return the shared Langfuse client (imported after .env is loaded)."""
    load_env()
    from langfuse import get_client

    return get_client()


@lru_cache(maxsize=1)
def get_callback_handler() -> CallbackHandler:
    """Return the LangChain callback handler that traces LLM generations."""
    load_env()
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def describe_documents(documents: list[Document]) -> list[dict[str, object]]:
    """Compact, JSON-safe view of retrieved chunks for a retriever observation."""
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": int(doc.metadata.get("page", 0)),
            "preview": doc.page_content[:200],
        }
        for doc in documents
    ]


@contextmanager
def answer_trace(
    question: str,
    *,
    technique: str,
    metadata: dict[str, object] | None = None,
) -> Iterator[Observation]:
    """Root span for one RAG request, tagged with the technique in use."""
    from langfuse import propagate_attributes

    langfuse = get_langfuse()
    with langfuse.start_as_current_observation(
        as_type="span", name=TRACE_NAME, input={"question": question}
    ) as span:
        with propagate_attributes(
            tags=["rag", f"technique:{technique}"], metadata=metadata or {}
        ):
            yield span


@contextmanager
def retrieval_trace(query: str) -> Iterator[Observation]:
    """Child ``retriever`` observation wrapping a retrieval step."""
    langfuse = get_langfuse()
    with langfuse.start_as_current_observation(
        as_type="retriever", name=RETRIEVAL_NAME, input={"query": query}
    ) as span:
        yield span


def flush() -> None:
    """Flush buffered traces; call before short-lived scripts exit."""
    if tracing_enabled():
        get_langfuse().flush()
