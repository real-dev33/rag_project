# app/tools/vector_search.py
import uuid
import logging
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorSearchProvider:
    def __init__(self, collection_name: str = "pdf_documents"):
        # Fix: create the client once and reuse it — don't recreate inside methods
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
            timeout=60.0
        )
        self.collection_name = collection_name
        self._init_collection()

    def _init_collection(self):
        """Create collection if it doesn't exist, with 384 dimensions."""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except (UnexpectedResponse, Exception):
            logger.info(f"Creating collection '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,  # 384 for MiniLM
                    distance=Distance.COSINE,
                )
            )

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Store document chunks and their embeddings in Qdrant.

        Fix 1: was async but QdrantClient is synchronous — removed async,
                caller uses run_in_executor for thread offload.
        Fix 2: was building points list then throwing it away — never called upsert.
        Fix 3: was recreating self.client at the end of the method (dead code).

        Args:
            chunks: List of chunk dicts with 'text', 'page', 'metadata' keys
            embeddings: Parallel list of embedding vectors
        """
        if not chunks:
            logger.warning("add_documents called with empty chunks list")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        points = []
        for chunk, emb in zip(chunks, embeddings):
            points.append(PointStruct(
                # Fix: use UUID so re-uploading the same document doesn't silently
                # collide with integer IDs from a prior upload
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text": chunk["text"],
                    "page": chunk.get("page"),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "chunk_index": chunk.get("chunk_index"),
                    "metadata": chunk.get("metadata", {}),
                }
            ))

        # Fix: actually upsert the points — this was missing entirely
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Upserted {len(points)} points into '{self.collection_name}'")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents by vector.

        Args:
            query_vector: Embedding of the query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of matching chunks with scores, including text, page, chunk_type, metadata, and similarity score
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "page": hit.payload.get("page"),
                "chunk_type": hit.payload.get("chunk_type"),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score,
            }
            for hit in response.points
        ]