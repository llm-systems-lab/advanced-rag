"""Reranking retrieval technique: retrieve candidates, rerank, then answer.

A wider set of candidate chunks is retrieved with dense search, re-scored with a
cross-encoder rerank model served by OpenRouter (``nvidia/nemotron-3.5-rerank-8b:free``
by default), and only the top-k reranked chunks are passed to the Gemini answer
model.
"""

from __future__ import annotations

import os

import requests
from langchain_core.documents import Document

from generation import Answer, generate
from ingestion import _require, load_env
from retrieval import TOP_K, retrieve_documents
from tracing import answer_trace, describe_documents, retrieval_trace

RERANK_CANDIDATES = 10
DEFAULT_RERANK_MODEL = "nvidia/nemotron-3.5-rerank-8b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RERANK_TIMEOUT = 60


def rerank_model() -> str:
    """Return the OpenRouter rerank model configured in .env."""
    load_env()
    return os.getenv("OPENROUTER_RERANK_MODEL", DEFAULT_RERANK_MODEL)


def rerank_documents(
    query: str, documents: list[Document], top_n: int = TOP_K
) -> list[Document]:
    """Re-score ``documents`` for ``query`` with the OpenRouter rerank model."""
    if not documents:
        return []
    load_env()
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/rerank",
        headers={
            "Authorization": f"Bearer {_require('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": rerank_model(),
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": top_n,
        },
        timeout=RERANK_TIMEOUT,
    )
    response.raise_for_status()
    results = response.json()["results"]
    return [documents[item["index"]] for item in results]


def retrieve_documents_rerank(
    query: str, k: int = TOP_K, candidates: int = RERANK_CANDIDATES
) -> list[Document]:
    """Retrieve ``candidates`` chunks, rerank them, and keep the top ``k``."""
    retrieved = retrieve_documents(query, k=candidates)
    return rerank_documents(query, retrieved, top_n=k)


def answer_rerank(question: str, k: int = TOP_K) -> Answer:
    """Answer the question using reranked context."""
    with answer_trace(
        question,
        technique="rerank",
        metadata={"k": k, "candidates": RERANK_CANDIDATES, "model": rerank_model()},
    ) as span:
        with retrieval_trace(question) as retriever:
            documents = retrieve_documents_rerank(question, k=k)
            retriever.update(output=describe_documents(documents))
        text = generate(question, documents)
        span.update(output={"answer": text})
    return Answer(text=text, sources=documents)
