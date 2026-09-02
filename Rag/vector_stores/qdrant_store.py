"""
Qdrant vector store implementation following the BaseVectorStore interface.
Supports both Qdrant Cloud (via url + api_key) and local Docker Qdrant instances.
"""

import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from .base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    def __init__(
        self,
        dimension: int = 384,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "pdf_documents",
        **kwargs
    ):
        """Initialize Qdrant vector store
        
        Args:
            dimension: Vector embedding dimension
            url: Qdrant cluster URL or local host URL
            api_key: Qdrant Cloud API Key (optional for local instance)
            collection_name: Qdrant collection name
        """
        super().__init__(dimension)
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("qdrant-client required. Install with: pip install qdrant-client")
        
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection(dimension, Distance, VectorParams)
        self._load_local_backup()

    def _get_backup_path(self) -> Path:
        cache_dir = Path(".rag_cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{self.collection_name}_backup.pkl"

    def _save_local_backup(self):
        import pickle
        try:
            with open(self._get_backup_path(), "wb") as f:
                pickle.dump((self.texts, self.embeddings, self.metadata), f)
        except Exception:
            pass

    def _load_local_backup(self):
        import pickle
        try:
            bpath = self._get_backup_path()
            if bpath.exists():
                with open(bpath, "rb") as f:
                    self.texts, self.embeddings, self.metadata = pickle.load(f)
        except Exception:
            pass

    def _ensure_collection(self, vector_size: int, Distance, VectorParams):
        """Ensure collection exists in Qdrant cluster"""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"[OK] Created Qdrant collection: {self.collection_name}")
            else:
                print(f"[INFO] Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"[WARN] Qdrant collection initialization warning: {e}")

    def add_vectors(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors and corresponding texts to Qdrant with metadata (idempotent using deterministic point IDs)"""
        from qdrant_client.models import PointStruct

        metadata = metadata or [{} for _ in texts]
        points = []
        
        for embedding, text, meta in zip(embeddings, texts, metadata):
            payload = {"text": text}
            payload.update(meta)
            
            # Generate deterministic UUID v5 to ensure idempotence across indexing runs
            doc_name = meta.get("source_file", "")
            page_num = meta.get("page_number", 0)
            chunk_idx = meta.get("chunk_index", 0)
            point_key = f"{doc_name}:{page_num}:{chunk_idx}:{text.strip()}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_key))
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
            
        # Batch upsert points to prevent WriteTimeout on large payloads
        batch_size = 100
        import time
        qdrant_failed = False
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            for attempt in range(3):
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    print(f"[WARN] Qdrant Cloud upsert error ({e}). Saving vectors locally to memory store.")
                    qdrant_failed = True
        
        # Track local state for BaseVectorStore interface compatibility & local fallback
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
        self._save_local_backup()
        if not qdrant_failed:
            print(f"[OK] Upserted {len(points)} vectors to Qdrant collection '{self.collection_name}' (in batches of {batch_size})")
        else:
            print(f"[OK] Indexed {len(points)} vectors into local memory store.")

    def _local_search(self, query_embedding: List[float], k: int = 5, return_metadata: bool = False) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        if not self.embeddings or not self.texts:
            return []
        import numpy as np
        vecs = np.array(self.embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-8)
        q_arr = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm
        sims = np.dot(vecs, q_arr)
        top_k = min(k, len(sims))
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        output = []
        for idx in top_indices:
            meta = self.metadata[idx] if return_metadata and idx < len(self.metadata) else None
            output.append((self.texts[idx], float(sims[idx]), meta))
        return output

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        return_metadata: bool = False
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Search Qdrant collection for similar vectors, falling back to local memory search if remote fails"""
        import time
        for attempt in range(3):
            try:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=k,
                    with_payload=True
                )
                
                output = []
                for hit in results.points:
                    payload = hit.payload or {}
                    text = payload.get("text", "")
                    score = float(hit.score)
                    meta = {k_key: v_val for k_key, v_val in payload.items() if k_key != "text"} if return_metadata else None
                    output.append((text, score, meta))
                return output
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                print(f"[WARN] Qdrant remote search error ({e}). Falling back to local memory search.")
                return self._local_search(query_embedding, k=k, return_metadata=return_metadata)
        return self._local_search(query_embedding, k=k, return_metadata=return_metadata)

    def save(self, filepath: str) -> None:
        """Data is automatically persisted on Qdrant server"""
        pass

    def load(self, filepath: str) -> None:
        """Data is managed server-side by Qdrant"""
        pass

    def clear(self) -> None:
        """Clear collection and recreate an empty collection on Qdrant server"""
        super().clear()
        import time
        for attempt in range(3):
            try:
                from qdrant_client.models import Distance, VectorParams
                collections = [c.name for c in self.client.get_collections().collections]
                if self.collection_name in collections:
                    self.client.delete_collection(self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"[OK] Cleared and recreated Qdrant collection: '{self.collection_name}'")
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                print(f"[WARN] Qdrant clear error: {e}")

    def size(self) -> int:
        """Return actual point count in Qdrant collection"""
        import time
        for attempt in range(3):
            try:
                res = self.client.count(collection_name=self.collection_name)
                return res.count
            except Exception:
                if attempt < 2:
                    time.sleep(1)
                    continue
                return super().size()
