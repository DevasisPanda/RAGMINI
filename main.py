"""
PDF RAG Application — CLI entry point.

Usage: python main.py
"""

import sys
import logging
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import Settings
from tinyrag import Provider, TinyRag, QueryResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def format_result(result: QueryResult) -> None:
    """Pretty-print a QueryResult with citations matching assignment requirements."""
    print(f"\nQuestion:\n{result.question}")
    print(f"\nAnswer:\n{result.answer}")

    if result.citations:
        print("\nSource:")
        for c in result.citations:  # Display all supporting citations
            print(f"{c.document_name}")
            print(f"Page {c.page_number}")
            print(f"Retrieved Text:")
            snippet = c.retrieved_text[:300].replace("\n", " ")
            print(f'"{snippet}..."')
            print()
    print("=" * 60)


def main() -> None:
    settings = Settings()

    if not settings.openrouter_api_key or settings.openrouter_api_key == "your-openrouter-api-key-here":
        logger.warning(
            "OPENROUTER_API_KEY is not set in .env or environment! "
            "Please set your key in .env file before running."
        )

    # Initialize provider pointing at OpenRouter
    provider = Provider(
        api_key=settings.openrouter_api_key,
        model=settings.llm_model,
        base_url="https://openrouter.ai/api/v1",
        embedding_model=settings.embedding_model,
        embedding_provider="local",
    )

    # Configure vector store parameters for Qdrant
    qdrant_config = {
        "url": settings.qdrant_url,
        "collection_name": settings.qdrant_collection,
    }
    if settings.qdrant_api_key:
        qdrant_config["api_key"] = settings.qdrant_api_key

    # Initialize TinyRag with Qdrant store
    rag = TinyRag(
        provider=provider,
        vector_store="qdrant",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        vector_store_config=qdrant_config,
        enable_cache=False,
    )

    # Look for PDF files in data/test_pdfs, data/pdfs, or current directory
    pdf_dir = Path("data/test_pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdf_files:
        pdf_dir = Path("data/pdfs")
        pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdf_files:
        pdf_files = list(Path(".").glob("*.pdf"))

    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF document(s) to index:")
        for f in pdf_files:
            logger.info(f"  - {f.name}")
        rag.add_pdf_documents(pdf_files)
    else:
        logger.warning("No PDF documents found in data/pdfs/ or current directory.")

    # Interactive Q&A loop
    print("\n" + "=" * 60)
    print("  PDF RAG Question-Answering System")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Enter Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            result = rag.ask(question, k=settings.top_k)
            format_result(result)
        except Exception as e:
            logger.error(f"Error answering question: {e}")


if __name__ == "__main__":
    main()
