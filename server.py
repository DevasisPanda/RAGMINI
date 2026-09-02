"""
AssignRAG Web API Server — FastAPI backend for portfolio chatbot integration.

Usage:
    python server.py
    or
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import Settings
from Rag import Provider, RagEngine, QueryResult
from controllers.provider_manager import ProviderManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")

# Global engine and manager state
rag_engine: Optional[RagEngine] = None
provider_manager: Optional[ProviderManager] = None
settings: Optional[Settings] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for initializing and shutting down the RAG backend."""
    global rag_engine, provider_manager, settings
    logger.info("Initializing AssignRAG Backend Server...")

    settings = Settings()

    # Initialize ProviderManager for LLM completions with OpenRouter -> Gemini failover
    provider_manager = ProviderManager(settings)

    # Initialize underlying Provider for local embeddings
    provider = Provider(
        api_key=settings.openrouter_api_key or "mock-key",
        model=settings.llm_model,
        base_url="https://openrouter.ai/api/v1",
        embedding_model=settings.embedding_model,
        embedding_provider="local",
    )
    # Hook chat_completion into ProviderManager for automatic failover
    provider.chat_completion = provider_manager.chat_completion

    # Vector store configuration
    qdrant_config = {
        "url": settings.qdrant_url,
        "collection_name": settings.qdrant_collection,
    }
    if settings.qdrant_api_key:
        qdrant_config["api_key"] = settings.qdrant_api_key

    # Initialize RAG Engine
    rag_engine = RagEngine(
        provider=provider,
        vector_store="qdrant",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        vector_store_config=qdrant_config,
        enable_cache=False,
    )

    logger.info(f"[OK] AssignRAG Engine initialized. Indexed vectors: {rag_engine.vector_store.size()}")
    yield
    logger.info("Shutting down AssignRAG Backend Server...")


app = FastAPI(
    title="AssignRAG Portfolio Chatbot API",
    description="REST API endpoint for integrating AssignRAG chatbot into web portfolios.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin requests from portfolio websites
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific domain (e.g., https://yourportfolio.com) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Models ---

class ChatRequest(BaseModel):
    message: Optional[str] = Field(None, description="User query message", example="Tell me about Devasis's experience.")
    question: Optional[str] = Field(None, description="Alternative field for user query", example="What are Devasis's skills?")
    top_k: Optional[int] = Field(10, description="Number of context chunks to retrieve", example=10)


class CitationResponse(BaseModel):
    document_name: str
    page_number: int
    retrieved_text: str


class ChatResponse(BaseModel):
    status: str = "success"
    question: str
    answer: str
    citations: List[CitationResponse]
    active_provider: str
    active_model: str


class HealthResponse(BaseModel):
    status: str
    indexed_vectors: int
    active_provider: str
    active_model: str


# --- API Routes ---

@app.get("/", tags=["Health"])
def root():
    return {
        "name": "AssignRAG Portfolio Chatbot API",
        "status": "online",
        "endpoints": {
            "chat": "POST /api/chat",
            "health": "GET /api/health",
            "index": "POST /api/index",
        }
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    if not rag_engine or not provider_manager:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")
    
    prov_name, model_name = provider_manager.get_active_provider_info()
    return HealthResponse(
        status="online",
        indexed_vectors=rag_engine.vector_store.size(),
        active_provider=prov_name,
        active_model=model_name,
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["Chatbot"])
def chat(request: ChatRequest):
    """Main REST endpoint for portfolio chatbot integration.
    
    Accepts user question and returns RAG answer with source citations.
    """
    if not rag_engine or not provider_manager:
        raise HTTPException(status_code=503, detail="RAG Engine is starting up. Try again shortly.")

    query = request.message or request.question
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Field 'message' or 'question' is required.")

    try:
        top_k = request.top_k or settings.top_k
        result: QueryResult = rag_engine.ask(query.strip(), k=top_k)

        citations_list = [
            CitationResponse(
                document_name=c.document_name,
                page_number=c.page_number,
                retrieved_text=c.retrieved_text[:300],
            )
            for c in result.citations
        ]

        prov_name, model_name = provider_manager.get_active_provider_info()

        return ChatResponse(
            status="success",
            question=result.question,
            answer=result.answer,
            citations=citations_list,
            active_provider=prov_name,
            active_model=model_name,
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index", tags=["Admin"])
def index_documents(background_tasks: BackgroundTasks):
    """Trigger re-indexing of PDFs in data/portfolio or data/pdfs directory."""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")

    pdf_dir = Path("data/portfolio")
    if not pdf_dir.exists():
        pdf_dir = Path("data/pdfs")

    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdf_files:
        raise HTTPException(status_code=404, detail="No PDF documents found in data/portfolio or data/pdfs")

    def _do_index():
        logger.info(f"Re-indexing {len(pdf_files)} PDF(s) into Qdrant...")
        rag_engine.add_pdf_documents(pdf_files, recreate_collection=True)
        logger.info(f"Indexing complete! Total vectors: {rag_engine.vector_store.size()}")

    background_tasks.add_task(_do_index)
    return {
        "status": "indexing_started",
        "documents_found": len(pdf_files),
        "files": [f.name for f in pdf_files]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
