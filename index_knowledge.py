"""
Portfolio Knowledge Base Indexer.

Run this script anytime you update professional.md or add new PDFs to data/portfolio/
to re-index your Qdrant vector database.

Usage:
    python index_knowledge.py
"""

import sys
import logging
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import Settings
from Rag import Provider, RagEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("indexer")


def main():
    settings = Settings()
    
    # Initialize Provider with local FastEmbed embeddings
    provider = Provider(
        api_key=settings.openrouter_api_key or "mock-key",
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
        embedding_provider="local",
    )

    qdrant_config = {
        "url": settings.qdrant_url,
        "collection_name": settings.qdrant_collection,
    }
    if settings.qdrant_api_key:
        qdrant_config["api_key"] = settings.qdrant_api_key

    rag = RagEngine(
        provider=provider,
        vector_store="qdrant",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        vector_store_config=qdrant_config,
        enable_cache=False,
    )

    # Collect all knowledge files: professional.md, PDFs in data/portfolio, etc.
    knowledge_files = []

    prof_md = Path("professional.md")
    if prof_md.exists():
        knowledge_files.append(prof_md)

    portfolio_dir = Path("data/portfolio")
    if portfolio_dir.exists():
        knowledge_files.extend(list(portfolio_dir.glob("*.pdf")))
        knowledge_files.extend(list(portfolio_dir.glob("*.md")))
        knowledge_files.extend(list(portfolio_dir.glob("*.txt")))

    # Fallback to test_pdfs or pdfs if no portfolio files exist yet
    if not knowledge_files:
        test_dir = Path("data/test_pdfs")
        if test_dir.exists():
            knowledge_files.extend(list(test_dir.glob("*.pdf")))

    if not knowledge_files:
        logger.error("No knowledge files found! Place professional.md in root or PDFs in data/portfolio/")
        return

    logger.info(f"Found {len(knowledge_files)} file(s) to index into Qdrant:")
    for f in knowledge_files:
        logger.info(f"  - {f.name}")

    # Process and index files
    all_chunks = []
    all_metadata = []

    for kfile in knowledge_files:
        ext = kfile.suffix.lower()
        if ext == ".pdf":
            from Rag.core.text_utils import extract_pdf_pages, chunk_text
            pages = extract_pdf_pages(str(kfile))
            for page_text, page_num in pages:
                chunks = chunk_text(page_text, settings.chunk_size, settings.chunk_overlap)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source_file": kfile.name,
                        "source_path": str(kfile),
                        "page_number": page_num,
                        "chunk_index": i,
                        "document_type": "pdf"
                    })
        else:
            from Rag.core.text_utils import extract_text, chunk_text
            text = extract_text(str(kfile))
            chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source_file": kfile.name,
                    "source_path": str(kfile),
                    "page_number": 1,
                    "chunk_index": i,
                    "document_type": ext.lstrip(".")
                })

    logger.info(f"Extracted {len(all_chunks)} chunks. Generating embeddings & uploading to Qdrant...")
    
    # Clear collection for fresh index run
    rag.clear_documents()
    
    embeddings = rag.provider.get_embeddings(all_chunks)
    rag.vector_store.add_vectors(embeddings, all_chunks, all_metadata)
    
    logger.info(f"[SUCCESS] Re-indexed {len(all_chunks)} chunks from {len(knowledge_files)} file(s) into Qdrant!")
    logger.info(f"Total vector count in Qdrant: {rag.vector_store.size()}")


if __name__ == "__main__":
    main()
