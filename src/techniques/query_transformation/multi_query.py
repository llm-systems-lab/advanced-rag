"""Multi-Query Retrieval with RAG Fusion.

The user question is expanded into several paraphrases with a fast generation
model (Groq). The original question plus the paraphrases are each sent to
Pinecone, and the ranked result lists are merged with Reciprocal Rank Fusion
(RRF) into a single ranking. The fused chunks, together with the original
question, are then passed to the Gemini answer model.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from generation import PROMPT, Answer, build_llm, format_context
from llm import build_groq_llm
from retrieval import TOP_K, retrieve_documents

RRF_K = 60
NUM_QUERIES = 3

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite search queries for technical product-documentation "
            "retrieval. Given a user question, generate "
            f"{NUM_QUERIES} alternative questions that capture the same "
            "information need from different angles: synonyms, rephrasing, and "
            "more specific technical wording. Return exactly "
            f"{NUM_QUERIES} questions, one per line, with no numbering, bullets, "
            "or extra text.",
        ),
        ("human", "{question}"),
    ]
)

_LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*")


def generate_multiple_queries(question: str, n: int = NUM_QUERIES) -> list[str]:
    """Use the Groq model to generate ``n`` paraphrased versions of a question."""
    chain = MULTI_QUERY_PROMPT | build_groq_llm() | StrOutputParser()
    raw = chain.invoke({"question": question})
    queries: list[str] = []
    for line in raw.splitlines():
        cleaned = _LIST_PREFIX.sub("", line).strip()
        if cleaned:
            queries.append(cleaned)
        if len(queries) == n:
            break
    return queries


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = TOP_K, rrf_k: int = RRF_K
) -> list[Document]:
    """Merge ranked result lists into one ranking with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, document in enumerate(ranked, start=1):
            key = document.metadata.get("chunk_id") or document.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            documents.setdefault(key, document)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [documents[key] for key in ordered[:k]]


def retrieve_documents_multi_query(
    question: str, queries: list[str] | None = None, k: int = TOP_K
) -> list[Document]:
    """Retrieve for the original plus generated queries, then fuse with RRF.

    Uses ``queries`` when provided, otherwise generates them with the LLM.
    """
    variants = queries or generate_multiple_queries(question)
    ranked_lists = [retrieve_documents(query, k=k) for query in [question, *variants]]
    return reciprocal_rank_fusion(ranked_lists, k=k)


def answer_multi_query(
    question: str, queries: list[str] | None = None, k: int = TOP_K
) -> Answer:
    """Answer the original question using RAG Fusion context."""
    documents = retrieve_documents_multi_query(question, queries=queries, k=k)
    chain = PROMPT | build_llm() | StrOutputParser()
    text = chain.invoke({"context": format_context(documents), "question": question})
    return Answer(text=text, sources=documents)
