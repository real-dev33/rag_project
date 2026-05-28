# app/tools/vector_search.py
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from app.core.config import settings

class VectorSearchProvider:
    def __init__(self, collection_name: str = "pdf_documents"):
        # Connect to Qdrant Cloud
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = collection_name
        self._init_collection()

    def _init_collection(self):
        """Create collection if it doesn't exist, with 384 dimensions."""
        try:
            self.client.get_collection(self.collection_name)
        except (UnexpectedResponse, Exception):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,                  # Matches MiniLM embedding dimension
                    distance=Distance.COSINE
                )
            )

    async def add_documents(self, chunks: list, embeddings: list):
        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=i,   # In production use a UUID or hash
                vector=emb,
                payload={
                    "text": chunk["text"],
                    "page": chunk.get("page"),
                    "metadata": chunk.get("metadata", {})
                }
            ))
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )