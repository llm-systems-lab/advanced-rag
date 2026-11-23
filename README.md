# Advanced RAG — Product Research Assistant

An experimental RAG application that answers technical questions about laptops using product specifications, user manuals, and hardware maintenance documentation.

The project explores advanced Retrieval-Augmented Generation techniques, document processing, vector search, and LLM evaluation.

## Features

- PDF document ingestion and processing
- Text extraction and document chunking
- Metadata extraction and storage
- Vector embeddings generation
- Semantic search using Pinecone
- Context-aware answer generation
- Source attribution with document names and page numbers
- Protection against prompt injection attempts

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM | Google Gemini |
| Embeddings | Ollama — all-minilm |
| Vector Database | Pinecone |
| LLM Framework | LangChain |
| Evaluation | Ragas |

## Project Structure

```text
advanced-rag/
├── src/
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── evaluation.py
│   └── main.py
├── data/
│   └── documents/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- uv
- Ollama
- Pinecone account
- Google AI API key

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/advanced-rag.git
cd advanced-rag

uv sync
```

### Configuration

Create a `.env` file based on `.env.example` and configure your API keys and model settings.

### Embedding Model

Pull the embedding model using Ollama:

```bash
ollama pull all-minilm
```

### Run

```bash
uv run python src/main.py
```
