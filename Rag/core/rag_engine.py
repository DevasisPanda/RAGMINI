"""
Main RagEngine class for Retrieval-Augmented Generation with multi-provider support
"""

from typing import Union, List, Optional, Dict, Any, Tuple
from pathlib import Path
import concurrent.futures
from threading import Lock
import time
import os

from .provider import Provider
from .text_utils import extract_text, chunk_text
from .code_parser import CodeParser
from .models import Citation, QueryResult
from ..vector_stores import (
    FaissVectorStore, 
    MemoryVectorStore, 
    PickleVectorStore, 
    ChromaVectorStore,
    QdrantVectorStore
)


class RagEngine:
    def __init__(
        self, 
        provider: Optional[Provider] = None, 
        vector_store: str = "faiss", 
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_config: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
        system_prompt: Optional[str] = None,
        enable_cache: bool = False,
        cache_dir: str = ".rag_cache"
    ):
        """Initialize RagEngine with optional provider and vector store
        
        Args:
            provider: Optional Provider instance for API calls. If None, uses local embeddings
            vector_store: Type of vector store ("faiss", "memory", "pickle", "chroma", "qdrant")
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Number of overlapping characters between chunks
            vector_store_config: Additional configuration for vector store
            max_workers: Maximum number of threads for parallel processing
            system_prompt: Custom system prompt for LLM chat
            enable_cache: Deprecated / unused
            cache_dir: Deprecated / unused
        """
        # Initialize provider or create default one
        if provider is None:
            self.provider = Provider()  # Uses default local embedding model
        else:
            self.provider = provider
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
        self._lock = Lock()  # For thread-safe operations
        
        # Set system prompt (assignment compliant default)
        self.system_prompt = system_prompt or (
            "You are a document question-answering assistant.\n\n"
            "RULES:\n"
            "1. Answer ONLY using the provided context from the retrieved documents.\n"
            "2. If the answer cannot be found in the context, respond exactly: "
            "\"The information is not available in the supplied documents.\"\n"
            "3. Do NOT fabricate or infer information beyond what is explicitly stated.\n"
            "4. Be concise and specific in your answers."
        )
        
        # Get embedding dimension from provider
        try:
            dimension = self.provider.get_embedding_dimension()
            print(f"[INFO] Embedding dimension: {dimension}")
        except Exception as e:
            print(f"[WARN] Could not determine embedding dimension: {e}")
            dimension = 384
        
        vector_store_config = vector_store_config or {}
        
        if vector_store == "faiss":
            self.vector_store = FaissVectorStore(dimension=dimension, **vector_store_config)
        elif vector_store == "memory":
            self.vector_store = MemoryVectorStore(dimension=dimension, **vector_store_config)
        elif vector_store == "pickle":
            self.vector_store = PickleVectorStore(dimension=dimension, **vector_store_config)
        elif vector_store == "chroma":
            self.vector_store = ChromaVectorStore(dimension=dimension, **vector_store_config)
        elif vector_store == "qdrant":
            self.vector_store = QdrantVectorStore(dimension=dimension, **vector_store_config)
        else:
            raise ValueError(f"Unsupported vector store: {vector_store}. "
                           f"Supported options: faiss, memory, pickle, chroma, qdrant")
    
    def _process_single_document(self, item: Union[str, Path]) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]], bool]:
        """Process a single document and return its chunks, embeddings, metadata, and cache status"""
        try:
            text = extract_text(item, show_progress=True)
            if text and text.strip():
                chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
                if chunks:
                    embeddings = self.provider.get_embeddings(chunks)
                    metadata = self._generate_metadata(item, chunks)
                    item_name = item if isinstance(item, str) and len(str(item)) < 50 else str(item)[:50] + '...'
                    print(f"[OK] Processed: {item_name} ({len(chunks)} chunks)")
                    return chunks, embeddings, metadata, False
                else:
                    print(f"[WARN] No chunks created from: {item}")
            else:
                print(f"[WARN] No text content found in: {item}")
        except Exception as e:
            print(f"[WARN] Failed to process {item}: {e}")
        return [], [], [], False

    def _generate_metadata(self, item: Union[str, Path], chunks: List[str]) -> List[Dict[str, Any]]:
        """Generate metadata for chunks from a document"""
        metadata_list = []
        
        # Determine if item is a file path or raw text
        is_file = isinstance(item, (str, Path)) and os.path.isfile(str(item))
        
        for i, chunk in enumerate(chunks):
            metadata = {
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            
            if is_file:
                file_path = Path(item)
                metadata.update({
                    'source_file': file_path.name,
                    'source_path': str(file_path),
                    'document_type': file_path.suffix.lower().lstrip('.') if file_path.suffix else 'txt',
                    'file_size': file_path.stat().st_size if file_path.exists() else None
                })
            else:
                # Raw text input
                metadata.update({
                    'source_file': None,
                    'source_path': None,
                    'document_type': 'text',
                    'is_raw_text': True
                })
            
            metadata_list.append(metadata)
        
        return metadata_list

    def add_documents(self, data: Union[str, Path, List[Union[str, Path]]], use_threading: bool = True) -> None:
        """Add documents from text, file paths, or mixed list
        
        Args:
            data: Can be:
                - Single file path: "document.pdf"
                - Single text string: "This is raw text"
                - List of files: ["doc1.pdf", "doc2.txt", "doc3.docx"]
                - List of texts: ["Text 1", "Text 2", "Text 3"]
                - Mixed list: ["document.pdf", "Raw text", "another.txt"]
            use_threading: Whether to use multithreading for processing multiple documents
        """
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        
        if isinstance(data, (list, tuple)):
            if use_threading and len(data) > 1:
                # Use multithreading for multiple documents
                print(f"Processing {len(data)} documents with multithreading (max_workers: {self.max_workers})...")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all documents for processing
                    future_to_item = {executor.submit(self._process_single_document, item): item for item in data}
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_item):
                        chunks, embeddings, metadata, _ = future.result()
                        if chunks:
                            with self._lock:  # Thread-safe addition
                                all_chunks.extend(chunks)
                                all_embeddings.extend(embeddings)
                                all_metadata.extend(metadata)
            else:
                # Sequential processing
                print(f"Processing {len(data)} documents sequentially...")
                for item in data:
                    chunks, embeddings, metadata, _ = self._process_single_document(item)
                    if chunks:
                        all_chunks.extend(chunks)
                        all_embeddings.extend(embeddings)
                        all_metadata.extend(metadata)
        else:
            # Handle single text or file path
            chunks, embeddings, metadata, _ = self._process_single_document(data)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            all_metadata.extend(metadata)
        
        if all_chunks:
            # Add to vector store (embeddings are already generated)
            with self._lock:
                self.vector_store.add_vectors(all_embeddings, all_chunks, all_metadata)
            
            # Summary
            total_chunks = len(all_chunks)
            print(f"[OK] Added {total_chunks} chunks to vector store")
        else:
            print("[WARN] No valid content found to add")
    
    def query(self, query: str, k: int = 5, return_scores: bool = True, return_metadata: bool = False) -> Union[List[str], List[Tuple[str, float]], List[Tuple[str, float, Optional[Dict[str, Any]]]]]:
        """Query the vector store without using LLM - just return similar chunks
        
        Args:
            query: Query string to search for
            k: Number of similar chunks to return
            return_scores: If True, return (text, score) tuples; if False, return just text
            return_metadata: If True, return (text, score, metadata) tuples
            
        Returns:
            List of similar text chunks, optionally with similarity scores and metadata
        """
        # Generate embedding for the query
        query_embedding = self.provider.get_embeddings([query])[0]
        
        # Search for relevant chunks
        results = self.vector_store.search(query_embedding, k=k, return_metadata=True)
        
        if return_metadata:
            return results  # Returns [(text, score, metadata), ...]
        elif return_scores:
            return [(text, score) for text, score, metadata in results]  # Returns [(text, score), ...]
        else:
            return [text for text, score, metadata in results]  # Returns [text, ...]
    
    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt for LLM chat
        
        Args:
            prompt: New system prompt to use for chat completions
        """
        self.system_prompt = prompt
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt
        
        Returns:
            Current system prompt string
        """
        return self.system_prompt
    
    def chat(self, query: str, k: int = 3) -> str:
        """Retrieve relevant chunks and generate an answer using LLM"""
        # Check if API key is available for chat completion
        if not self.provider.api_key:
            raise ValueError(
                "No API key provided. Chat functionality requires an API key. "
                "Initialize RagEngine with: Provider(api_key='your-key') or use query() method for similarity search only."
            )
        
        # Generate embedding for the query
        query_embedding = self.provider.get_embeddings([query])[0]
        
        # Search for relevant chunks
        results = self.vector_store.search(query_embedding, k=k)
        
        if not results:
            context = "No relevant documents found."
        else:
            # Combine retrieved chunks as context
            context_chunks = [text for text, score, *_ in results]
            context = "\n\n".join(context_chunks)
        
        # Create messages for chat completion
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
        
        # Generate response using the provider
        response = self.provider.chat_completion(messages)
        return response
    
    def add_pdf_documents(
        self, 
        pdf_paths: Union[str, Path, List[Union[str, Path]]], 
        recreate_collection: bool = True
    ) -> None:
        """Ingest PDF files with page-level metadata tracking for citations.
        
        Args:
            pdf_paths: Path or list of paths to PDF documents
            recreate_collection: If True (default), clears the existing vector collection
                before indexing so that only the current PDF set is searchable.
        """
        from .text_utils import extract_pdf_pages, chunk_text

        if isinstance(pdf_paths, (str, Path)):
            pdf_paths = [pdf_paths]

        if recreate_collection:
            coll_name = getattr(self.vector_store, 'collection_name', 'vector_store')
            print(f"[INFO] Recreating collection '{coll_name}' for fresh indexing run...")
            self.clear_documents()

        all_chunks = []
        all_metadata = []

        for pdf_path in pdf_paths:
            pdf_path_obj = Path(pdf_path)
            if not pdf_path_obj.exists():
                print(f"[WARN] File not found: {pdf_path}")
                continue

            pages = extract_pdf_pages(str(pdf_path_obj))
            print(f"[INFO] Processing {pdf_path_obj.name}: {len(pages)} pages extracted")

            doc_chunks_count = 0
            for page_text, page_num in pages:
                page_chunks = chunk_text(page_text, self.chunk_size, self.chunk_overlap)
                for i, chunk in enumerate(page_chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source_file": pdf_path_obj.name,
                        "source_path": str(pdf_path_obj),
                        "page_number": page_num,
                        "chunk_index": i,
                        "document_type": "pdf",
                    })
                    doc_chunks_count += 1
            print(f"[OK] Processed {pdf_path_obj.name}: {doc_chunks_count} chunks across {len(pages)} pages")

        if all_chunks:
            print(f"\n[INFO] PDF parsing complete: {len(all_chunks)} total chunks extracted across {len(pdf_paths)} document(s).")
            all_embeddings = self.provider.get_embeddings(all_chunks)
            with self._lock:
                self.vector_store.add_vectors(all_embeddings, all_chunks, all_metadata)
            print(f"[OK] Successfully indexed {len(all_chunks)} chunks from {len(pdf_paths)} PDF(s) into vector store!\n")
        else:
            print("[WARN] No valid PDF content extracted")

    def _expand_query(self, query: str) -> str:
        """Internal query expansion for legal/semantic search."""
        q_lower = query.lower()
        expansions = []

        if "basu" in q_lower or "d.k. basu" in q_lower or "dk basu" in q_lower:
            expansions.append("directions guidelines memo of arrest identification name tags medical examination custodial torture procedural safeguards lockup death")
        if "ashwani" in q_lower or "ashwani kumar" in q_lower:
            expansions.append("torture human rights act preventive measures custodial violence")
        if "githa" in q_lower or "hariharan" in q_lower or "githa hariharan" in q_lower:
            expansions.append("natural guardian mother father custody minority guardianship")
        if "chandra kumar" in q_lower or "l. chandra kumar" in q_lower:
            expansions.append("tribunal judicial review high court supreme court constitutional validity")
        if "arrest memo" in q_lower or "memo" in q_lower:
            expansions.append("memo of arrest witness signature time place magistrate copy")
        if "safeguard" in q_lower or "safeguards" in q_lower or "arrest" in q_lower:
            expansions.append("arrest detention medical examination bail grounds of arrest rights")

        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query

    def _get_doc_boost(self, query: str, text: str, metadata: dict) -> float:
        """Document-aware retrieval score boost if query mentions case/doc title, and content boost for substantive holding terms."""
        q_lower = query.lower()
        doc_name = (metadata.get("source_file") or "").lower()

        boost = 0.0
        doc_patterns = {
            "basu": "d_k_basu",
            "d.k. basu": "d_k_basu",
            "dk basu": "d_k_basu",
            "ashwani": "ashwani_kumar",
            "githa": "githa_hariharan",
            "hariharan": "githa_hariharan",
            "chandra kumar": "l_chandra_kumar",
            "keisham": "keisham",
            "meghachandra": "keisham",
            "kihoto": "kihoto",
            "hollohan": "kihoto",
            "lalita": "lalita_kumari",
            "kailash": "kailash_nath"
        }

        has_explicit_doc = False
        for keyword, pattern in doc_patterns.items():
            if keyword in q_lower:
                has_explicit_doc = True
                if pattern in doc_name:
                    boost += 0.20
                else:
                    boost -= 0.10

        if has_explicit_doc:
            text_lower = text.lower()
            substantive_keywords = [
                "guidelines", "directions", "memo of arrest", "attested by",
                "medical examination", "identification and name tags",
                "interrogation", "held that", "court held", "safeguards"
            ]
            if any(kw in text_lower for kw in substantive_keywords):
                boost += 0.10

        return boost

    def _is_comparison_query(self, query: str, detected_docs: List[Tuple[str, str]]) -> bool:
        """Detect if query expresses a comparison intent between documents."""
        q_lower = query.lower()
        explicit_keywords = [
            "compare", "comparison", "compare with", "difference", "different",
            "differ", "similarity", "similarities", "contrast", "contrasting"
        ]
        import re
        for kw in explicit_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', q_lower):
                return True

        # Check if 'vs' or 'versus' is used between 2 or more distinct detected case documents
        if len(detected_docs) >= 2:
            vs_keywords = ["versus", "vs", "v."]
            for kw in vs_keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', q_lower):
                    return True

        return False

    def _detect_referenced_documents(self, query: str) -> List[Tuple[str, str]]:
        """Detect supported case/document names in query.
        Returns list of (display_name, pattern_string) tuples without duplicates.
        """
        q_lower = query.lower()
        doc_patterns = [
            ("D.K. Basu", "d_k_basu", ["basu", "d.k. basu", "dk basu"]),
            ("Ashwani Kumar", "ashwani_kumar", ["ashwani", "ashwani kumar"]),
            ("Githa Hariharan", "githa_hariharan", ["githa", "hariharan", "githa hariharan"]),
            ("L. Chandra Kumar", "l_chandra_kumar", ["chandra kumar", "l. chandra kumar", "l chandra kumar"]),
            ("Keisham Meghachandra", "keisham", ["keisham", "meghachandra"]),
            ("Kihoto Hollohan", "kihoto", ["kihoto", "hollohan"]),
            ("Lalita Kumari", "lalita_kumari", ["lalita", "lalita kumari"]),
            ("Kailash Nath", "kailash_nath", ["kailash", "kailash nath"])
        ]

        detected = []
        seen_patterns = set()
        for display_name, pattern, keywords in doc_patterns:
            if any(kw in q_lower for kw in keywords):
                if pattern not in seen_patterns:
                    seen_patterns.add(pattern)
                    detected.append((display_name, pattern))

        return detected

    def _get_single_doc_boost(self, target_pattern: str, text: str, metadata: dict) -> float:
        """Score boost targeting a specific document pattern for balanced comparison retrieval."""
        doc_name = (metadata.get("source_file") or "").lower()
        boost = 0.0
        if target_pattern in doc_name:
            boost += 0.25
            text_lower = text.lower()
            substantive_keywords = [
                "guidelines", "directions", "memo of arrest", "attested by",
                "medical examination", "identification and name tags",
                "interrogation", "held that", "court held", "safeguards",
                "judgment", "held", "article", "section", "guardian", "tribunal"
            ]
            if any(kw in text_lower for kw in substantive_keywords):
                boost += 0.10
        else:
            boost -= 0.15
        return boost

    def ask(self, query: str, k: int = 10, verbose: bool = False) -> QueryResult:
        """RAG query with structured citations. Returns QueryResult object.
        
        Args:
            query: User question to search and answer
            k: Number of top relevant context chunks to retrieve (default: 10)
            verbose: If True, prints raw search results, similarity scores, candidate count, and LLM context
        """
        if not self.provider.api_key:
            raise ValueError(
                "No API key provided. Chat functionality requires an API key. "
                "Initialize Provider(api_key='your-key', base_url='https://openrouter.ai/api/v1')"
            )

        detected_docs = self._detect_referenced_documents(query)
        is_comp = self._is_comparison_query(query, detected_docs)

        if verbose:
            if is_comp:
                print("\n[DEBUG] Comparison Mode: YES")
                if detected_docs:
                    print("[DEBUG] Detected documents:")
                    for disp, _ in detected_docs:
                        print(f"  - {disp}")
                else:
                    print("[DEBUG] Detected documents: None")
            else:
                print("\n[DEBUG] Comparison Mode: NO")

        # ----------------------------------------------------
        # MULTI-DOCUMENT COMPARISON RETRIEVAL PATH
        # ----------------------------------------------------
        if is_comp and len(detected_docs) >= 2:
            k_per_doc = max(3, k // len(detected_docs))
            per_doc_chunks = {}
            doc_stats = []
            has_insufficient_doc = False

            for disp_name, pattern in detected_docs:
                doc_query = self._expand_query(disp_name)
                combined_q = f"{query} {disp_name}"
                queries_to_embed = [doc_query, combined_q]

                embeddings = self.provider.get_embeddings(queries_to_embed)
                primary_emb = embeddings[0]

                raw_candidates = self.vector_store.search(primary_emb, k=30, return_metadata=True)
                if len(embeddings) > 1:
                    exp_candidates = self.vector_store.search(embeddings[1], k=30, return_metadata=True)
                    seen_texts = {}
                    for txt, sc, meta in raw_candidates + exp_candidates:
                        if txt not in seen_texts or sc > seen_texts[txt][1]:
                            seen_texts[txt] = (txt, sc, meta)
                    raw_candidates = list(seen_texts.values())

                def doc_rank_score(item):
                    text, score, meta = item
                    m = meta or {}
                    doc_boost = self._get_single_doc_boost(pattern, text, m)
                    is_cover_header = text.startswith("Copyright @ Manupatra") or ("MANU/SC/" in text[:120] and "JUDGMENT" not in text[:120])
                    header_penalty = 0.15 if is_cover_header else 0.0
                    return score + doc_boost - header_penalty

                sorted_doc_candidates = sorted(raw_candidates, key=doc_rank_score, reverse=True)

                # Deduplicate and page-diversify for this doc
                seen_keys = set()
                page_counts = {}
                selected_for_doc = []

                for text, score, metadata in sorted_doc_candidates:
                    meta = metadata or {}
                    source_file = (meta.get("source_file") or "").lower()
                    if pattern not in source_file:
                        continue  # Ensure chunks come from the target document!

                    page_num = meta.get("page_number", 0)
                    dedup_key = (source_file, page_num, text.strip())

                    if dedup_key not in seen_keys and page_counts.get(page_num, 0) < 2:
                        seen_keys.add(dedup_key)
                        page_counts[page_num] = page_counts.get(page_num, 0) + 1
                        selected_for_doc.append((text, score, metadata))

                    if len(selected_for_doc) >= k_per_doc:
                        break

                max_doc_sim = max((s for _, s, _ in selected_for_doc), default=0.0)
                if not selected_for_doc or max_doc_sim < 0.35:
                    has_insufficient_doc = True

                per_doc_chunks[disp_name] = selected_for_doc
                doc_stats.append((disp_name, len(selected_for_doc)))

            if verbose:
                print("[DEBUG] Retrieved:")
                for disp_name, count in doc_stats:
                    print(f"  - {count} chunks from {disp_name}")

            # Merge all chunks
            merged_chunks = []
            for disp_name, chunks in per_doc_chunks.items():
                merged_chunks.extend(chunks)

            if verbose:
                print(f"[DEBUG] Merged:")
                print(f"  - {len(merged_chunks)} final chunks")

            # Check unknown condition for comparison
            if has_insufficient_doc or not merged_chunks:
                return QueryResult(
                    question=query,
                    answer="The supplied documents do not contain sufficient information to compare both requested cases.",
                    citations=[],
                    is_answerable=False,
                )

            selected_chunks = merged_chunks
            comparison_system_prompt = (
                "You are a legal document question-answering assistant specializing in comparative analysis.\n\n"
                "RULES:\n"
                "1. Answer ONLY using the provided context from the retrieved documents.\n"
                "2. Compare each requested case separately based strictly on the context provided.\n"
                "3. For each case include:\n"
                "   - Constitutional Issue / Legal Question\n"
                "   - Court Reasoning & Key Observations\n"
                "   - Final Holding / Directions\n"
                "4. Provide explicit sections for:\n"
                "   - Key Similarities\n"
                "   - Key Differences\n"
                "5. Do NOT fabricate, infer, or use outside legal knowledge beyond what is explicitly stated in the context.\n"
                "6. If the supplied documents do not contain sufficient information to compare both requested cases, respond exactly: "
                "\"The supplied documents do not contain sufficient information to compare both requested cases.\""
            )

            citations = []
            context_parts = []
            for i, (text, score, metadata) in enumerate(selected_chunks, 1):
                meta = metadata or {}
                doc_name = meta.get("source_file", "Unknown")
                page_num = meta.get("page_number", 0)

                citations.append(Citation(
                    document_name=doc_name,
                    page_number=page_num,
                    retrieved_text=text,
                    similarity_score=float(score)
                ))
                context_parts.append(
                    f"[Source {i}] Document: {doc_name}, Page {page_num}:\n{text}"
                )

            context = "\n\n".join(context_parts)

            if verbose:
                print(f"\n[DEBUG] Final Comparison Prompt Context:\n{context}\n")

            messages = [
                {"role": "system", "content": comparison_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Context from documents:\n{context}\n\n"
                        f"Question: {query}\n\n"
                        "Instructions: Provide a detailed comparison between the requested cases using ONLY the context above. "
                        "Address the constitutional issues, legal questions, court reasoning, observations, and holdings for each case, "
                        "followed by explicit similarities and differences. "
                        "If the context lacks sufficient information for either case, state: 'The supplied documents do not contain sufficient information to compare both requested cases.'"
                    ),
                },
            ]

            answer = self.provider.chat_completion(messages, temperature=0.2)

            unanswerable_phrase = "sufficient information to compare"
            if unanswerable_phrase in answer.lower() or "information is not available" in answer.lower():
                return QueryResult(
                    question=query,
                    answer="The supplied documents do not contain sufficient information to compare both requested cases.",
                    citations=[],
                    is_answerable=False,
                )

            return QueryResult(
                question=query,
                answer=answer,
                citations=citations,
                is_answerable=True,
            )

        if is_comp and len(detected_docs) == 0:
            if verbose:
                print("[DEBUG] Comparison Mode requested but no matching legal documents detected in query. Marking UNKNOWN.")
            return QueryResult(
                question=query,
                answer="The information is not available in the supplied documents.",
                citations=[],
                is_answerable=False,
            )

        if is_comp and len(detected_docs) == 1:
            if verbose:
                print("[DEBUG] Comparison Mode requested but only 1 matching document detected in query. Marking UNKNOWN.")
            return QueryResult(
                question=query,
                answer="The supplied documents do not contain sufficient information to compare both requested cases.",
                citations=[],
                is_answerable=False,
            )

        # ----------------------------------------------------
        # STANDARD RETRIEVAL PATH (UNCHANGED)
        # ----------------------------------------------------
        # 1. Query Expansion (Requirement 3)
        expanded_q = self._expand_query(query)

        # Generate embeddings for query and expanded query
        queries_to_embed = [query]
        if expanded_q != query:
            queries_to_embed.append(expanded_q)

        embeddings = self.provider.get_embeddings(queries_to_embed)
        primary_emb = embeddings[0]

        # 2. Internal Candidate Pool Retrieval (Requirement 1 & 6: 25-30 candidates)
        candidate_pool_size = 30
        raw_candidates = self.vector_store.search(primary_emb, k=candidate_pool_size, return_metadata=True)

        if len(embeddings) > 1:
            expanded_candidates = self.vector_store.search(embeddings[1], k=candidate_pool_size, return_metadata=True)
            # Merge candidates keeping highest similarity score per unique text
            seen_texts = {}
            for txt, sc, meta in raw_candidates + expanded_candidates:
                if txt not in seen_texts or sc > seen_texts[txt][1]:
                    seen_texts[txt] = (txt, sc, meta)
            raw_candidates = list(seen_texts.values())

        if verbose:
            print(f"\n[DEBUG] Query: '{query}'")
            if expanded_q != query:
                print(f"[DEBUG] Expanded Query: '{expanded_q}'")
            print(f"[DEBUG] Internal Candidate Pool Count: {len(raw_candidates)}")
            print("[DEBUG] Raw Similarity Scores & Candidate Chunks:")
            for idx, (txt, sc, meta) in enumerate(raw_candidates, 1):
                m = meta or {}
                snip = txt[:90].replace('\n', ' ')
                print(f"  [{idx}] Raw Score: {sc:.4f} | Document: {m.get('source_file')} | Page: {m.get('page_number')} | Snippet: {snip}...")

        if not raw_candidates:
            return QueryResult(
                question=query,
                answer="The information is not available in the supplied documents.",
                citations=[],
                is_answerable=False,
            )

        # 3. Document-aware boosting & Cover-header penalty (Requirement 2)
        def final_rank_score(item):
            text, score, meta = item
            m = meta or {}
            doc_boost = self._get_doc_boost(query, text, m)
            is_cover_header = text.startswith("Copyright @ Manupatra") or ("MANU/SC/" in text[:120] and "JUDGMENT" not in text[:120])
            header_penalty = 0.15 if is_cover_header else 0.0
            return score + doc_boost - header_penalty

        sorted_candidates = sorted(raw_candidates, key=final_rank_score, reverse=True)

        # 4. Filter Candidate Pool: Page Diversity, Document Diversity & Exact Chunk Deduplication (Requirement 1)
        seen_keys = set()
        page_counts = {}
        selected_chunks = []

        # Pass 1: Max 2 chunks per page to ensure broad coverage
        for text, score, metadata in sorted_candidates:
            meta = metadata or {}
            doc_name = meta.get("source_file", "Unknown")
            page_num = meta.get("page_number", 0)
            dedup_key = (doc_name, page_num, text.strip())

            if dedup_key not in seen_keys and page_counts.get(page_num, 0) < 2:
                seen_keys.add(dedup_key)
                page_counts[page_num] = page_counts.get(page_num, 0) + 1
                selected_chunks.append((text, score, metadata))

            if len(selected_chunks) >= k:
                break

        # Pass 2 Fallback if page diversity yield is less than k
        if len(selected_chunks) < k:
            for text, score, metadata in sorted_candidates:
                meta = metadata or {}
                doc_name = meta.get("source_file", "Unknown")
                page_num = meta.get("page_number", 0)
                dedup_key = (doc_name, page_num, text.strip())

                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    selected_chunks.append((text, score, metadata))

                if len(selected_chunks) >= k:
                    break

        # 5. Better Unknown Detection (Requirement 4)
        # Check best similarity score in selected chunks
        max_sim = max((s for _, s, _ in selected_chunks), default=0.0)
        if not selected_chunks or max_sim < 0.35:
            if verbose:
                print(f"[DEBUG] Maximum similarity score ({max_sim:.4f}) is below threshold 0.35. Marking query UNKNOWN.")
            return QueryResult(
                question=query,
                answer="The information is not available in the supplied documents.",
                citations=[],
                is_answerable=False,
            )

        if verbose:
            print(f"\n[DEBUG] Selected Chunks Count: {len(selected_chunks)}")
            print("[DEBUG] Selected Chunks for Prompt:")
            for idx, (txt, sc, meta) in enumerate(selected_chunks, 1):
                m = meta or {}
                print(f"  [{idx}] Score: {sc:.4f} | Document: {m.get('source_file')} | Page: {m.get('page_number')}")

        citations = []
        context_parts = []
        for i, (text, score, metadata) in enumerate(selected_chunks, 1):
            meta = metadata or {}
            doc_name = meta.get("source_file", "Unknown")
            page_num = meta.get("page_number", 0)

            citations.append(Citation(
                document_name=doc_name,
                page_number=page_num,
                retrieved_text=text,
                similarity_score=float(score)
            ))
            context_parts.append(
                f"[Source {i}] Document: {doc_name}, Page {page_num}:\n{text}"
            )

        context = "\n\n".join(context_parts)

        if verbose:
            print(f"\n[DEBUG] Final Prompt Context:\n{context}\n")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context from documents:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    "Instructions: Read the context carefully. If the answer to the question can be found or derived from any of the context sources above, provide a clear, accurate answer citing the details. If the answer cannot be found in the provided context, respond exactly: 'The information is not available in the supplied documents.'"
                ),
            },
        ]

        answer = self.provider.chat_completion(messages, temperature=0.2)

        # 6. Post-generation Unknown check
        unanswerable_phrase = "the information is not available in the supplied documents"
        if unanswerable_phrase in answer.lower() or "information is not available" in answer.lower():
            return QueryResult(
                question=query,
                answer="The information is not available in the supplied documents.",
                citations=[],
                is_answerable=False,
            )

        return QueryResult(
            question=query,
            answer=answer,
            citations=citations,
            is_answerable=True,
        )
    
    def query_structured(self, query: str, k: int = 5) -> QueryResult:
        """Query with structured output including sources and citations"""
        results = self.query(query, k=k, return_metadata=True)
        citations = []
        for text, score, metadata in results:
            meta = metadata or {}
            citations.append(Citation(
                document_name=meta.get("source_file", "Unknown"),
                page_number=meta.get("page_number", 0),
                retrieved_text=text,
                similarity_score=float(score)
            ))
        return QueryResult(
            question=query,
            answer="",
            citations=citations,
            is_answerable=bool(citations)
        )
    
    def chat_structured(self, query: str, k: int = 3) -> QueryResult:
        """Enhanced chat with structured output including sources and citations"""
        return self.ask(query, k=k)
    
    def get_similar_chunks(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Get chunks similar to the provided text (alias for query with scores)"""
        return self.query(text, k=k, return_scores=True)
    
    def search_documents(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Tuple[str, float]]:
        """Search documents with optional minimum similarity score filtering"""
        results = self.query(query, k=k, return_scores=True)
        return [(text, score) for text, score in results if score >= min_score]
    
    def get_all_chunks(self) -> List[str]:
        """Get all stored text chunks"""
        return self.vector_store.texts.copy()
    
    def get_chunk_count(self) -> int:
        """Get the number of stored chunks"""
        return self.vector_store.size()
    
    def clear_documents(self) -> None:
        """Clear all stored documents and embeddings"""
        self.vector_store.clear()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider configuration"""
        return {
            "embedding_provider": getattr(self.provider, 'embedding_provider', 'unknown'),
            "embedding_model": getattr(self.provider, 'embedding_model', 'unknown'),
            "chat_model": getattr(self.provider, 'model', 'unknown'),
            "embedding_dimension": getattr(self.provider, 'embedding_dimension', 'unknown'),
            "base_url": getattr(self.provider, 'base_url', ''),
            "ollama_base_url": getattr(self.provider, 'ollama_base_url', ''),
            "has_api_key": bool(getattr(self.provider, 'api_key', None))
        }
    
    def save_vector_store(self, filepath: str) -> None:
        """Save the vector store to disk"""
        self.vector_store.save(filepath)
    
    def load_vector_store(self, filepath: str) -> None:
        """Load the vector store from disk"""
        self.vector_store.load(filepath)
    
    def add_code_file(self, file_path: str) -> None:
        """Add a single code file by parsing and indexing its functions
        
        Args:
            file_path: Path to the code file to process
        """
        from pathlib import Path
        
        file_path_obj = Path(file_path)
        
        # Check if file exists
        if not file_path_obj.exists():
            print(f"⚠ File not found: {file_path}")
            return
        
        # Check if it's a file (not directory)
        if not file_path_obj.is_file():
            print(f"⚠ Path is not a file: {file_path}")
            return
        
        # Check if it's a supported code file
        if not CodeParser.is_code_file(str(file_path)):
            print(f"⚠ Unsupported file type: {file_path}")
            return
        
        print(f"Processing code file: {file_path}")
        
        # Parse the file
        functions = CodeParser.parse_file(str(file_path))
        
        if functions and functions[0]['type'] != 'error':
            # Format functions for embedding
            formatted_chunks = []
            for func in functions:
                chunk = f"File: {func['file']}\nLanguage: {func['language']}\nType: {func['type']}\nName: {func['name']}\nCode:\n{func['content']}"
                formatted_chunks.append(chunk)
            
            print(f"Generating embeddings for {len(formatted_chunks)} code functions...")
            
            # Generate embeddings for all chunks
            embeddings = self.provider.get_embeddings(formatted_chunks)
            
            # Add to vector store
            self.vector_store.add_vectors(embeddings, formatted_chunks)
            
            print(f"✓ Added {len(formatted_chunks)} code functions from {file_path_obj.name} to vector store")
        else:
            print(f"⚠ No valid code functions found in: {file_path}")
    
    def add_codebase(self, path: str, recursive: bool = True, use_threading: bool = True) -> None:
        """Add codebase from directory or single file by parsing and indexing code functions
        
        Args:
            path: Path to directory containing code files OR path to a single code file
            recursive: Whether to scan subdirectories recursively (only applies to directories)
            use_threading: Whether to use multithreading for processing (only applies to directories with multiple files)
        
        Examples:
            # Process a single code file
            rag.add_codebase("path/to/my_file.py")
            
            # Process a directory
            rag.add_codebase("path/to/project/")
            
            # Process directory non-recursively
            rag.add_codebase("path/to/project/", recursive=False)
        """
        from pathlib import Path
        
        path_obj = Path(path)
        
        # Check if path exists
        if not path_obj.exists():
            print(f"⚠ Path not found: {path}")
            return
        
        # Handle single file
        if path_obj.is_file():
            self.add_code_file(str(path))
            return
        
        # Handle directory
        if not path_obj.is_dir():
            print(f"⚠ Path is neither a file nor a directory: {path}")
            return
        
        # Scan directory for code files
        code_files = CodeParser.scan_directory(path, recursive=recursive)
        
        if not code_files:
            print(f"⚠ No code files found in: {path}")
            return
        
        print(f"Found {len(code_files)} code files in {path}")
        
        all_functions = []
        
        if use_threading and len(code_files) > 1:
            # Use multithreading for multiple files
            print(f"Processing {len(code_files)} code files with multithreading...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files for processing
                future_to_file = {executor.submit(CodeParser.parse_file, file_path): file_path for file_path in code_files}
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    functions = future.result()
                    if functions and functions[0]['type'] != 'error':
                        with self._lock:  # Thread-safe addition
                            all_functions.extend(functions)
        else:
            # Sequential processing
            print(f"Processing {len(code_files)} code files sequentially...")
            for file_path in code_files:
                functions = CodeParser.parse_file(file_path)
                if functions and functions[0]['type'] != 'error':
                    all_functions.extend(functions)
        
        if all_functions:
            # Format functions for embedding
            formatted_chunks = []
            for func in all_functions:
                chunk = f"File: {func['file']}\nLanguage: {func['language']}\nType: {func['type']}\nName: {func['name']}\nCode:\n{func['content']}"
                formatted_chunks.append(chunk)
            
            print(f"Generating embeddings for {len(formatted_chunks)} code functions...")
            
            # Generate embeddings for all chunks
            embeddings = self.provider.get_embeddings(formatted_chunks)
            
            # Add to vector store (thread-safe)
            with self._lock:
                self.vector_store.add_vectors(embeddings, formatted_chunks)
            
            print(f"✓ Added {len(formatted_chunks)} code functions to vector store")
        else:
            print("⚠ No valid code functions found to add")
    
    def search_code(self, query: str, k: int = 5, language: Optional[str] = None, min_score: float = 0.0) -> List[Tuple[str, float]]:
        """Search for code functions
        
        Args:
            query: Search query
            k: Number of results to return
            language: Filter by programming language (optional)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (code_chunk, score) tuples
        """
        # Generate embedding for the query
        query_embedding = self.provider.get_embeddings([query])[0]
        
        # Search for relevant chunks
        results = self.vector_store.search(query_embedding, k=k * 2)  # Get more results for filtering
        
        # Filter by language if specified
        if language:
            filtered_results = []
            for text, score in results:
                if f"Language: {language.lower()}" in text.lower() and score >= min_score:
                    filtered_results.append((text, score))
                    if len(filtered_results) >= k:
                        break
            return filtered_results[:k]
        
        # Filter by minimum score
        return [(text, score) for text, score in results if score >= min_score][:k]
    
    def get_function_by_name(self, function_name: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for functions by name
        
        Args:
            function_name: Name of function to search for
            k: Number of results to return
            
        Returns:
            List of (code_chunk, score) tuples
        """
        # Generate embedding for the function name
        query_embedding = self.provider.get_embeddings([function_name])[0]
        
        # Search for relevant chunks
        results = self.vector_store.search(query_embedding, k=k)
        
        # Filter to only include exact name matches
        exact_matches = []
        for text, score in results:
            if f"Name: {function_name}" in text:
                exact_matches.append((text, score))
        
        return exact_matches[:k]