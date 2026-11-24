"""Hybrid retrieval technique: answer using dense + sparse (alpha-weighted) context."""

from __future__ import annotations

from generation import Answer, generate
from retrieval import TOP_K, default_alpha, retrieve_documents_hybrid
from tracing import answer_trace, describe_documents, retrieval_trace


def answer_hybrid(
    question: str, k: int = TOP_K, alpha: float | None = None
) -> Answer:
    """Answer the question using hybrid-retrieved context."""
    resolved_alpha = default_alpha() if alpha is None else alpha
    with answer_trace(
        question, technique="hybrid", metadata={"k": k, "alpha": resolved_alpha}
    ) as span:
        with retrieval_trace(question) as retriever:
            documents = retrieve_documents_hybrid(
                question, k=k, alpha=resolved_alpha
            )
            retriever.update(output=describe_documents(documents))
        text = generate(question, documents)
        span.update(output={"answer": text})
    return Answer(text=text, sources=documents)
