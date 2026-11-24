"""Retrieval layer: dense and hybrid (dense + sparse) search over Pinecone."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.db_data.index import Index
from pinecone.db_data.types import SparseVectorTypedDict

from ingestion import (
    embed_sparse,
    get_client,
    get_embeddings,
    index_name,
    l2_normalize,
    load_env,
    require_ready_index,
    wait_ready,
)

TOP_K = 4
DEFAULT_ALPHA = 0.9


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Return the shared Ollama embedding model (built once)."""
    load_env()
    return get_embeddings()


@lru_cache(maxsize=1)
def get_index() -> Index:
    """Connect to the raw Pinecone index once (used for dense + hybrid queries)."""
    name = index_name()
    pc = get_client()
    require_ready_index(pc, name)
    wait_ready(pc, name)
    return pc.Index(name)


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    """Return the LangChain vector store backed by the shared index."""
    return PineconeVectorStore(index=get_index(), embedding=get_embedding_model())


@lru_cache(maxsize=1)
def default_alpha() -> float:
    """Dense/sparse balance, overridable via ``HYBRID_ALPHA`` in .env."""
    load_env()
    return float(os.getenv("HYBRID_ALPHA", str(DEFAULT_ALPHA)))


def embed_query(query: str) -> list[float]:
    """Embed a single query string with the configured Ollama model."""
    return get_embedding_model().embed_query(query)


def retrieve_documents(query: str, k: int = TOP_K) -> list[Document]:
    """Retrieve the top-k most relevant chunks for a query from Pinecone."""
    return get_vector_store().similarity_search(query, k=k)


def hybrid_score_norm(
    dense: list[float], sparse: SparseVectorTypedDict, alpha: float
) -> tuple[list[float], SparseVectorTypedDict]:
    """Apply the convex combination ``alpha*dense + (1-alpha)*sparse``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return (
        [value * alpha for value in dense],
        {
            "indices": sparse["indices"],
            "values": [value * (1 - alpha) for value in sparse["values"]],
        },
    )


def retrieve_documents_hybrid(
    query: str, k: int = TOP_K, alpha: float | None = None
) -> list[Document]:
    """Retrieve with dense + sparse signals combined by alpha weighting."""
    alpha = default_alpha() if alpha is None else alpha
    dense = l2_normalize(get_embedding_model().embed_query(query))
    sparse = embed_sparse([query], "query")[0]
    hdense, hsparse = hybrid_score_norm(dense, sparse, alpha)
    result = get_index().query(
        vector=hdense, sparse_vector=hsparse, top_k=k, include_metadata=True
    )

    documents: list[Document] = []
    for match in result.matches:
        metadata = dict(match.metadata or {})
        text = metadata.pop("text", "")
        documents.append(Document(page_content=text, metadata=metadata))
    return documents
