"""HyDE (Hypothetical Document Embeddings): answer hypothetically, then retrieve.

The HyDE technique asks a fast generation model (Groq) to draft a hypothetical
answer passage for the user question. That passage is embedded and used for
retrieval, because it is closer to the target documents than the bare question.
The retrieved chunks are then passed, together with the original question, to
the Gemini answer model.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from generation import Answer, generate
from llm import build_groq_llm
from retrieval import TOP_K, retrieve_documents
from tracing import (
    answer_trace,
    describe_documents,
    get_callback_handler,
    retrieval_trace,
)

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a technical documentation writer. Write a short, factual "
            "passage that directly answers the question, as it might appear in a "
            "product manual or specification sheet. Do not mention uncertainty or "
            "that the passage is hypothetical. Return only the passage.",
        ),
        ("human", "{question}"),
    ]
)


def generate_hypothetical_answer(question: str) -> str:
    """Use the Groq model to draft a hypothetical answer passage."""
    chain = HYDE_PROMPT | build_groq_llm() | StrOutputParser()
    return chain.invoke(
        {"question": question}, config={"callbacks": [get_callback_handler()]}
    ).strip()


def retrieve_documents_hyde(
    question: str, hypothetical_answer: str | None = None, k: int = TOP_K
) -> list[Document]:
    """Embed the hypothetical answer and retrieve the most relevant chunks.

    Uses ``hypothetical_answer`` when provided, otherwise generates it with the LLM.
    """
    passage = hypothetical_answer or generate_hypothetical_answer(question)
    return retrieve_documents(passage, k=k)


def answer_hyde(
    question: str, hypothetical_answer: str | None = None, k: int = TOP_K
) -> Answer:
    """Answer the original question with context retrieved via HyDE."""
    with answer_trace(question, technique="hyde", metadata={"k": k}) as span:
        with retrieval_trace(question) as retriever:
            documents = retrieve_documents_hyde(
                question, hypothetical_answer=hypothetical_answer, k=k
            )
            retriever.update(output=describe_documents(documents))
        text = generate(question, documents)
        span.update(output={"answer": text})
    return Answer(text=text, sources=documents)
