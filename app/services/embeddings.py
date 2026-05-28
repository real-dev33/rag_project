# app/services/embedding.py
from typing import List
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Load the model once, globally, when the module is imported.
# Model downloads automatically on first use and caches locally.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text using a free local model.
    """
    cleaned = text.replace("\n", " ").strip()
    embedding = model.encode(cleaned, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings(
    texts: List[str],
    batch_size: int = 32
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts with batching.
    This is much faster than calling generate_embedding() in a loop.

    Args:
        texts: List of chunk texts
        batch_size: Number of texts to process at once

    Returns:
        List of embedding vectors, same order as input
    """
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    logger.info(f"Generating embeddings for {len(texts)} texts (batch size {batch_size})")

    embeddings = model.encode(
        cleaned,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.tolist()