"""
Application configuration using Pydantic Settings.
"""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM Primary — OpenRouter
    openrouter_api_key: Optional[str] = None
    llm_model: str = "google/gemma-2-9b-it:free"

    # LLM Secondary — Google Gemini (fallback)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    provider_failover: bool = True

    # Vector Store — Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "pdf_documents"

    # Embedding Model
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking & Retrieval Parameters
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 10

    # Debug & Logging
    debug_mode: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
