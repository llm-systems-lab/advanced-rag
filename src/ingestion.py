"""Ingestion pipeline: load PDFs, split into chunks, embed (dense + sparse) and
upsert into the shared Pinecone index.

The single index stores a dense vector (Ollama) and a sparse vector (Pinecone's
hosted ``pinecone-sparse-english-v0``) per chunk, which enables both dense-only
and hybrid retrieval.

Usage:
    uv run python src/ingestion.py --reset   # re-ingest the configured sources
    uv run python src/ingestion.py --wipe    # clear the whole index first
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.db_data.types import SparseVectorTypedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_METRIC = "dotproduct"
DEFAULT_INDEX_NAME = "advanced-rag"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SPARSE_MODEL = "pinecone-sparse-english-v0"
SPARSE_BATCH = 96
DENSE_BATCH = 64
UPSERT_BATCH = 100
READY_TIMEOUT = 300


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Copy .env.example to .env and fill it in."
        )
    return value


@lru_cache(maxsize=1)
def load_env() -> None:
    """Load .env once for the lifetime of the process."""
    load_dotenv(PROJECT_ROOT / ".env")


def index_name() -> str:
    """Return the configured Pinecone index name (the shared hybrid index)."""
    load_env()
    return os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)


@lru_cache(maxsize=1)
def get_client() -> Pinecone:
    """Return a shared Pinecone client."""
    load_env()
    return Pinecone(api_key=_require("PINECONE_API_KEY"))


def load_documents(directory: Path = DOCUMENTS_DIR) -> list[Document]:
    """Load every PDF under ``directory`` (recursively) into LangChain documents."""
    pdf_paths = sorted(directory.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {directory}")

    documents: list[Document] = []
    for path in pdf_paths:
        pages = PyPDFLoader(str(path)).load()
        for page in pages:
            page.metadata["source"] = str(path.relative_to(PROJECT_ROOT))
            page.metadata["category"] = path.parent.name
        documents.extend(pages)
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks with a stable ``chunk_id``."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        key = f"{chunk.metadata.get('source')}:{chunk.metadata.get('page')}:{index}"
        chunk.metadata["chunk_id"] = hashlib.sha1(key.encode()).hexdigest()
    return chunks


def get_embeddings() -> OllamaEmbeddings:
    """Return the Ollama embedding model configured in .env."""
    return OllamaEmbeddings(model=_require("OLLAMA_EMBEDDING_MODEL"))


def ensure_index(
    pc: Pinecone, name: str, dimension: int, metric: str = PINECONE_METRIC
) -> None:
    """Create the serverless dense index if it does not exist yet.

    The shared index stores dense + sparse vectors, which requires
    ``metric="dotproduct"``. An existing index whose metric or dimension does
    not match is rejected so an incompatible index is never reused silently.
    """
    if pc.has_index(name):
        description = pc.describe_index(name)
        if description.dimension != dimension or description.metric != metric:
            raise RuntimeError(
                f"Index '{name}' already exists with metric={description.metric!r} "
                f"and dimension={description.dimension}, but this pipeline requires "
                f"metric={metric!r} and dimension={dimension}. Recreate the index or "
                f"point PINECONE_INDEX_NAME at a compatible one."
            )
        return
    pc.create_index(
        name=name,
        vector_type="dense",
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )


def require_ready_index(pc: Pinecone, name: str) -> None:
    """Fail fast on the read path when the shared index is missing or empty."""
    if not pc.has_index(name):
        raise RuntimeError(
            f"Pinecone index '{name}' does not exist. Run ingestion first: "
            f"uv run python src/ingestion.py"
        )
    description = pc.describe_index(name)
    if (
        description.metric != PINECONE_METRIC
        or description.dimension != EMBEDDING_DIMENSION
    ):
        raise RuntimeError(
            f"Pinecone index '{name}' has metric={description.metric!r} and "
            f"dimension={description.dimension}, but this pipeline requires "
            f"metric={PINECONE_METRIC!r} and dimension={EMBEDDING_DIMENSION}. "
            f"Recreate the index or point PINECONE_INDEX_NAME at a compatible one."
        )
    if not pc.Index(name).describe_index_stats().total_vector_count:
        raise RuntimeError(
            f"Pinecone index '{name}' is empty. Run ingestion first: "
            f"uv run python src/ingestion.py"
        )


def wait_ready(pc: Pinecone, name: str, timeout: int = READY_TIMEOUT) -> None:
    """Block until the index is ready, failing instead of hanging forever."""
    deadline = time.monotonic() + timeout
    while not pc.describe_index(name).status.ready:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Pinecone index '{name}' was not ready within {timeout}s"
            )
        time.sleep(1)


def l2_normalize(values: list[float]) -> list[float]:
    """Scale a vector to unit L2 norm so dense and sparse scores are comparable."""
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def embed_sparse(texts: list[str], input_type: str) -> list[SparseVectorTypedDict]:
    """Embed texts into unit-norm sparse vectors with Pinecone's hosted model."""
    embeddings: list[SparseVectorTypedDict] = []
    for start in range(0, len(texts), SPARSE_BATCH):
        batch = texts[start : start + SPARSE_BATCH]
        result = get_client().inference.embed(
            model=SPARSE_MODEL,
            inputs=batch,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        embeddings.extend(
            {
                "indices": item.sparse_indices,
                "values": l2_normalize(item.sparse_values),
            }
            for item in result
        )
    return embeddings


def ingest(reset: bool = False, wipe: bool = False) -> None:
    """Load, chunk, embed (dense + sparse) and upsert into the shared index.

    ``reset`` deletes only the chunks whose ``source`` is being re-ingested;
    ``wipe`` deletes every vector in the index and must be requested explicitly.
    """
    load_env()
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages -> {len(chunks)} chunks")

    embeddings = get_embeddings()
    pc = get_client()
    name = index_name()
    ensure_index(pc, name, EMBEDDING_DIMENSION)
    wait_ready(pc, name)
    index = pc.Index(name)

    if wipe:
        index.delete(delete_all=True)
        print("Cleared all vectors from the index")
    elif reset:
        sources = sorted({chunk.metadata.get("source", "unknown") for chunk in chunks})
        index.delete(filter={"source": {"$in": sources}})
        print(f"Cleared existing vectors for {len(sources)} source(s)")

    for start in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[start : start + UPSERT_BATCH]
        texts = [chunk.page_content for chunk in batch]

        dense_vectors: list[list[float]] = []
        for dense_start in range(0, len(texts), DENSE_BATCH):
            dense_vectors.extend(
                l2_normalize(vector)
                for vector in embeddings.embed_documents(
                    texts[dense_start : dense_start + DENSE_BATCH]
                )
            )
        sparse_vectors = embed_sparse(texts, "passage")

        records = [
            {
                "id": chunk.metadata["chunk_id"],
                "values": dense,
                "sparse_values": sparse,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "unknown"),
                    "page": int(chunk.metadata.get("page", 0)),
                    "category": chunk.metadata.get("category", ""),
                    "chunk_id": chunk.metadata["chunk_id"],
                },
            }
            for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors)
        ]
        index.upsert(vectors=records)
    print(f"Upserted {len(chunks)} records into '{name}'")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest documents into the shared Pinecone index"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete chunks for the sources being ingested before upserting",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete every vector in the index before upserting (destructive)",
    )
    args = parser.parse_args()
    ingest(reset=args.reset, wipe=args.wipe)


if __name__ == "__main__":
    main()
