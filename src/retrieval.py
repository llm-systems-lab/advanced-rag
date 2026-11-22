"""Retrieval layer: embed a query and fetch the most relevant chunks from Pinecone."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from ingestion import EMBEDDING_DIMENSION, PROJECT_ROOT, ensure_index, get_embeddings

TOP_K = 4


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Return the shared Ollama embedding model (built once)."""
    return get_embeddings()


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    """Connect to the Pinecone index once and reuse the connection."""
    load_dotenv(PROJECT_ROOT / ".env")
    index_name = os.getenv("PINECONE_INDEX_NAME", "advanced-rag")
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing required environment variable 'PINECONE_API_KEY'. "
            "Copy .env.example to .env and fill it in."
        )

    pc = Pinecone(api_key=api_key)
    ensure_index(pc, index_name, EMBEDDING_DIMENSION)
    return PineconeVectorStore(index=pc.Index(index_name), embedding=get_embedding_model())


def embed_query(query: str) -> list[float]:
    """Embed a single query string with the configured Ollama model."""
    return get_embedding_model().embed_query(query)


def retrieve_documents(query: str, k: int = TOP_K) -> list[Document]:
    """Retrieve the top-k most relevant chunks for a query from Pinecone."""
    return get_vector_store().similarity_search(query, k=k)
