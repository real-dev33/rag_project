# test_qdrant.py (run from project root)
from app.tools.vector_search import VectorSearchProvider
from app.core.config import settings

print(f"Connecting to {settings.qdrant_url}...")
provider = VectorSearchProvider(collection_name="test_collection")
print("Collection ready.")
# Optional: try inserting a dummy point
import numpy as np
dummy_vec = np.zeros(384).tolist()
provider.client.upsert(
    collection_name="test_collection",
    points=[{"id": 1, "vector": dummy_vec, "payload": {"text": "test"}}]
)
print("Dummy point inserted, cloud connection works!")