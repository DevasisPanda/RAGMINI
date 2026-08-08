"""
Qdrant vector store implementation following the BaseVectorStore interface.
Supports both Qdrant Cloud (via url + api_key) and local Docker Qdrant instances.
"""

import uuid
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
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        # Track local state for BaseVectorStore interface compatibility
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
        print(f"[OK] Upserted {len(points)} vectors to Qdrant collection '{self.collection_name}' (in batches of {batch_size})")

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        return_metadata: bool = False
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Search Qdrant collection for similar vectors"""
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
            print(f"[WARN] Qdrant search error: {e}")
            return []

    def save(self, filepath: str) -> None:
        """Data is automatically persisted on Qdrant server"""
        pass

    def load(self, filepath: str) -> None:
        """Data is managed server-side by Qdrant"""
        pass

    def clear(self) -> None:
        """Clear collection and recreate an empty collection on Qdrant server"""
        super().clear()
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
        except Exception as e:
            print(f"[WARN] Qdrant clear error: {e}")

    def size(self) -> int:
        """Return actual point count in Qdrant collection"""
        try:
            res = self.client.count(collection_name=self.collection_name)
            return res.count
        except Exception:
            return super().size()
