"""
Backend Controller providing a thread-safe bridge between the GUI and the TinyRag engine.
"""

import queue
import threading
import logging
from pathlib import Path
from typing import List, Callable, Any, Optional, Tuple

from config import Settings
from tinyrag import Provider, TinyRag, QueryResult
from .provider_manager import ProviderManager
from .file_manager import FileManager

logger = logging.getLogger(__name__)


class BackendController:
    """Orchestrates TinyRag operations on background threads with queue-based UI updates."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.file_manager = FileManager()
        self.provider_manager = ProviderManager(settings)
        self.message_queue: queue.Queue = queue.Queue()
        self._rag: Optional[TinyRag] = None
        self._is_busy: bool = False
        self._lock = threading.Lock()

        # Initialize backend engine
        self._init_rag_engine()

    def _init_rag_engine(self) -> None:
        """Initialize Provider and TinyRag instance."""
        try:
            # Initialize provider pointing at OpenRouter / local embeddings
            provider = Provider(
                api_key=self.settings.openrouter_api_key,
                model=self.settings.llm_model,
                base_url="https://openrouter.ai/api/v1",
                embedding_model=self.settings.embedding_model,
                embedding_provider="local",
            )

            # Delegate chat completion calls to ProviderManager for automatic failover
            provider.chat_completion = self.provider_manager.chat_completion

            # Configure vector store parameters for Qdrant
            qdrant_config = {
                "url": self.settings.qdrant_url,
                "collection_name": self.settings.qdrant_collection,
            }
            if self.settings.qdrant_api_key:
                qdrant_config["api_key"] = self.settings.qdrant_api_key

            # Initialize TinyRag
            self._rag = TinyRag(
                provider=provider,
                vector_store="qdrant",
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                vector_store_config=qdrant_config,
                enable_cache=False,
            )
            logger.info("[OK] TinyRag engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TinyRag engine: {e}")
            self.post_event("error", f"TinyRag Init Error: {e}")

    def is_busy(self) -> bool:
        """Check if a background operation is currently running."""
        return self._is_busy

    def post_event(self, event_type: str, data: Any) -> None:
        """Thread-safe event poster to GUI message queue."""
        self.message_queue.put((event_type, data))

    def index_pdfs_async(self, pdf_paths: List[Path], recreate_collection: bool = True) -> None:
        """Start indexing PDFs on a background worker thread."""
        if self._is_busy:
            self.post_event("warning", "Operation already in progress. Please wait.")
            return

        self._is_busy = True
        thread = threading.Thread(
            target=self._index_worker,
            args=(pdf_paths, recreate_collection),
            daemon=True
        )
        thread.start()

    def _index_worker(self, pdf_paths: List[Path], recreate_collection: bool) -> None:
        """Worker thread for PDF indexing."""
        try:
            self.post_event("status", ("indexing", "Initializing indexing pipeline..."))
            self.post_event("progress", 0.10)

            if not self._rag:
                self._init_rag_engine()

            if not pdf_paths:
                self.post_event("warning", "No PDF files selected for indexing.")
                self.post_event("status", ("ready", "Ready (No PDFs)"))
                return

            self.post_event("progress", 0.30)
            self.post_event("status", ("indexing", f"Indexing {len(pdf_paths)} PDF(s)..."))

            # Run PDF indexing pipeline
            str_paths = [str(p) for p in pdf_paths]
            self._rag.add_pdf_documents(str_paths, recreate_collection=recreate_collection)

            chunk_count = self._rag.get_chunk_count()
            self.post_event("progress", 1.0)
            self.post_event("status", ("ready", f"Ready ({chunk_count} chunks indexed)"))
            self.post_event("index_complete", {
                "chunk_count": chunk_count,
                "file_count": len(pdf_paths)
            })
            logger.info(f"[OK] Indexing completed: {chunk_count} vectors active in Qdrant.")

        except Exception as e:
            logger.error(f"Indexing error: {e}")
            self.post_event("error", f"Indexing failed: {e}")
            self.post_event("status", ("error", "Indexing error"))
        finally:
            self._is_busy = False

    def ask_question_async(self, query: str) -> None:
        """Start RAG question answering on a background worker thread."""
        if self._is_busy:
            self.post_event("warning", "Operation already in progress. Please wait.")
            return

        if not query or not query.strip():
            return

        self._is_busy = True
        thread = threading.Thread(
            target=self._ask_worker,
            args=(query.strip(),),
            daemon=True
        )
        thread.start()

    def _ask_worker(self, query: str) -> None:
        """Worker thread for RAG question answering."""
        try:
            self.post_event("status", ("thinking", "Searching documents & generating answer..."))

            if not self._rag:
                self._init_rag_engine()

            # Execute RAG query
            result: QueryResult = self._rag.ask(query, k=self.settings.top_k, verbose=self.settings.debug_mode)

            self.post_event("answer", result)
            self.post_event("status", ("ready", "Ready"))

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            self.post_event("error", f"Query Error: {e}")
            self.post_event("status", ("error", "Query error"))
        finally:
            self._is_busy = False

    def get_provider_display_info(self) -> Tuple[str, str]:
        """Get (provider_name, model_name) for display in GUI status bar."""
        return self.provider_manager.get_active_provider_info()
