"""Ingestion pipeline: load PDFs, split into chunks, embed and upsert to Pinecone."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Copy .env.example to .env and fill it in."
        )
    return value


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


def ensure_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    """Create the serverless Pinecone index if it does not exist yet."""
    if pc.has_index(index_name):
        return
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )


def ingest(reset: bool = False) -> PineconeVectorStore:
    """Run the full ingestion pipeline and return the populated vector store."""
    load_dotenv(PROJECT_ROOT / ".env")
    index_name = os.getenv("PINECONE_INDEX_NAME", "advanced-rag")

    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages -> {len(chunks)} chunks")

    embeddings = get_embeddings()
    pc = Pinecone(api_key=_require("PINECONE_API_KEY"))
    ensure_index(pc, index_name, EMBEDDING_DIMENSION)
    vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)

    if reset:
        vector_store.delete(delete_all=True)
        print("Cleared existing vectors from the index")

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)
    print(f"Upserted {len(chunks)} chunks into '{index_name}'")
    return vector_store
