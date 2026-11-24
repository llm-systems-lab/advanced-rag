"""Step-back prompting: generalise the query, retrieve twice, merge, then answer.

The step-back technique asks the LLM to rewrite the user question into a broader
"step-back" question, retrieves context for both the original and the step-back
question, merges the results (removing duplicates) and answers the original
question with that combined context.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from generation import Answer, build_llm, generate
from retrieval import TOP_K, retrieve_documents
from tracing import (
    answer_trace,
    describe_documents,
    get_callback_handler,
    retrieval_trace,
)

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at world knowledge. Given a specific question, "
            "write a more general step-back question that captures the broader "
            "concept or principle behind it. Return only the question.",
        ),
        ("human", "{question}"),
    ]
)


def generate_step_back_question(question: str) -> str:
    """Use the LLM to generate a broader step-back version of the question."""
    chain = STEP_BACK_PROMPT | build_llm() | StrOutputParser()
    return chain.invoke(
        {"question": question}, config={"callbacks": [get_callback_handler()]}
    ).strip()


def _dedupe(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    merged: list[Document] = []
    for document in documents:
        key = document.metadata.get("chunk_id") or document.page_content
        if key in seen:
            continue
        seen.add(key)
        merged.append(document)
    return merged


def retrieve_documents_step_back(
    question: str, step_back_question: str | None = None, k: int = TOP_K
) -> list[Document]:
    """Retrieve for the original and step-back questions, then merge and dedupe.

    Uses ``step_back_question`` when provided, otherwise generates it with the LLM.
    """
    step_back = step_back_question or generate_step_back_question(question)
    original_hits = retrieve_documents(question, k=k)
    step_back_hits = retrieve_documents(step_back, k=k)
    return _dedupe(original_hits + step_back_hits)


def answer_step_back(
    question: str, step_back_question: str | None = None, k: int = TOP_K
) -> Answer:
    """Answer the original question using the merged step-back context."""
    with answer_trace(question, technique="step_back", metadata={"k": k}) as span:
        with retrieval_trace(question) as retriever:
            documents = retrieve_documents_step_back(
                question, step_back_question=step_back_question, k=k
            )
            retriever.update(output=describe_documents(documents))
        text = generate(question, documents)
        span.update(output={"answer": text})
    return Answer(text=text, sources=documents)
