"""Contextual Compression: extract only question-relevant passages from chunks.

Unlike reranking (which re-scores and reorders whole chunks), contextual
compression keeps the retrieval order and runs each chunk through an LLM
extractor that returns only the sentences/passages relevant to the question,
copied verbatim from the original chunk. Chunks with no relevant content are
dropped. The compressed passages keep their source metadata, so the Gemini
answer model still sees document names and page numbers.

The extraction is purely extractive: the model copies text, it does not
paraphrase or add facts, which protects factual accuracy.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from generation import Answer, build_llm, generate
from retrieval import TOP_K, retrieve_documents
from tracing import (
    answer_trace,
    describe_documents,
    get_callback_handler,
    retrieval_trace,
)

NONE_TOKEN = "NONE"

COMPRESSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You extract relevant information from a single retrieved document "
                "chunk.\n"
                "Given an excerpt and a question, return ONLY the sentences or "
                "passages from the excerpt that are relevant to answering the "
                "question.\n"
                "Rules:\n"
                "- Copy the relevant text verbatim from the excerpt. Do not "
                "paraphrase, summarize, rewrite, reorder, or add information.\n"
                "- Use ONLY text that already appears in the excerpt.\n"
                "- Preserve exact values, units, model names, and terminology.\n"
                "- Include enough surrounding text to keep each passage "
                "self-contained and understandable.\n"
                "- If no part of the excerpt is relevant, return exactly "
                f"{NONE_TOKEN} and nothing else."
            ),
        ),
        ("human", "Question:\n{question}\n\nExcerpt:\n{chunk}"),
    ]
)


@lru_cache(maxsize=1)
def _compression_chain() -> Runnable[dict[str, str], str]:
    """Build the extractor chain once and reuse it across chunks."""
    return COMPRESSION_PROMPT | build_llm() | StrOutputParser()


def compress_document(question: str, document: Document) -> Document | None:
    """Extract the relevant passages of ``document`` for ``question``.

    Returns a new document with the extracted text and the same metadata, or
    ``None`` when nothing in the chunk is relevant.
    """
    extracted = (
        _compression_chain()
        .invoke(
            {"question": question, "chunk": document.page_content},
            config={"callbacks": [get_callback_handler()]},
        )
        .strip()
    )
    if not extracted or extracted.upper().strip(" .") == NONE_TOKEN:
        return None
    return Document(page_content=extracted, metadata=dict(document.metadata))


def compress_documents(question: str, documents: list[Document]) -> list[Document]:
    """Compress each chunk to its question-relevant passages, dropping empties."""
    compressed: list[Document] = []
    for document in documents:
        reduced = compress_document(question, document)
        if reduced is not None:
            compressed.append(reduced)
    return compressed


def retrieve_documents_compress(question: str, k: int = TOP_K) -> list[Document]:
    """Retrieve the top-``k`` chunks, then extract their relevant passages."""
    documents = retrieve_documents(question, k=k)
    return compress_documents(question, documents)


def answer_compress(question: str, k: int = TOP_K) -> Answer:
    """Answer the question using compressed (passage-level) context."""
    with answer_trace(question, technique="compression", metadata={"k": k}) as span:
        with retrieval_trace(question) as retriever:
            documents = retrieve_documents_compress(question, k=k)
            retriever.update(output=describe_documents(documents))
        text = generate(question, documents)
        span.update(output={"answer": text})
    return Answer(text=text, sources=documents)
