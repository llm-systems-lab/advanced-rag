"""Shared LLM builders for models other than the Gemini generation model."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr

from ingestion import PROJECT_ROOT


def build_groq_llm(temperature: float = 0.0) -> ChatGroq:
    """Return the Groq chat model configured in .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    if not api_key or not model:
        raise RuntimeError(
            "Missing required environment variables 'GROQ_API_KEY' and "
            "'GROQ_MODEL'. Copy .env.example to .env and fill them in."
        )
    return ChatGroq(model=model, api_key=SecretStr(api_key), temperature=temperature)
