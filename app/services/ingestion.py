# app/services/ingestion.py
import asyncio
import logging
from pathlib import Path

# Import your pipeline
from .text_extraction.pipeline import ExtractionPipeline

# Import the free embedding function you just created
# Use absolute import to avoid relative import resolution issues
from services.embeddings import generate_embeddings

# Import your vector database wrapper (create this next if you haven't)
from tools.vector_search import VectorSearchProvider

# Import config for Qdrant URL
from core.config import settings

logger = logging.getLogger(__name__)

# Instantiate the pipeline once (it's stateless enough to reuse)
pipeline = ExtractionPipeline()


async def process_pdf(file_path: str, task_id: str):
    """
    Background task that:
    1. Extracts text/chunks from PDF using your pipeline
    2. Generates embeddings for all chunks
    3. Stores chunks + embeddings in Qdrant
    """
    try:
        logger.info(f"[{task_id}] Starting PDF processing: {file_path}")

        # Step 1: Run the extraction pipeline (synchronous, so we offload)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, pipeline.extract, file_path
        )

        if not result.is_successful:
            logger.error(f"[{task_id}] Extraction failed or no text extracted")
            # Update task status in DB to 'failed' (if you have a tasks table)
            return

        chunks = result.chunks  # List[Dict] with 'text', 'page', 'metadata'
        logger.info(f"[{task_id}] Extracted {len(chunks)} chunks")

        # Step 2: Get embeddings for all chunk texts (free, local)
        texts = [c['text'] for c in chunks]
        embeddings = await loop.run_in_executor(
            None, generate_embeddings, texts
        )

        # Step 3: Store in Qdrant
        vector_provider = VectorSearchProvider(collection_name="pdf_documents")
        await vector_provider.add_documents(chunks, embeddings)

        logger.info(f"[{task_id}] Successfully stored {len(chunks)} chunks in Qdrant")
        # Update task status to 'completed' in DB here

    except Exception as e:
        logger.error(f"[{task_id}] Error processing PDF: {str(e)}")
        # Update task status to 'failed'