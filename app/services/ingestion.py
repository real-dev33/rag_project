# app/services/ingestion.py
import asyncio
import logging
import os

from app.services.text_extraction.pipeline import ExtractionPipeline
from app.services.embeddings import generate_embeddings
from app.tools.vector_search import VectorSearchProvider
from app.core.config import settings

logger = logging.getLogger(__name__)

pipeline = ExtractionPipeline()


def process_pdf(file_path: str, task_id: str):
    """
    Background task (sync entry point) dispatched by FastAPI BackgroundTasks.

    FastAPI's BackgroundTasks runs in a thread pool without an event loop,
    so this must be a plain sync function. Async work is done inside a
    fresh event loop spun up here.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_process_pdf_async(file_path, task_id))
    finally:
        loop.close()
        # Clean up the delete=False temp file created in main.py
        try:
            os.unlink(file_path)
            logger.debug(f"[{task_id}] Deleted temp file: {file_path}")
        except OSError as e:
            logger.warning(f"[{task_id}] Could not delete temp file {file_path}: {e}")


async def _process_pdf_async(file_path: str, task_id: str):
    """Internal async implementation of the PDF processing pipeline."""
    try:
        logger.info(f"[{task_id}] Starting PDF processing: {file_path}")

        loop = asyncio.get_running_loop()

        # Step 1: Extract (CPU-bound — offload to thread)
        result = await loop.run_in_executor(None, pipeline.extract, file_path)

        if not result.is_successful:
            logger.error(f"[{task_id}] Extraction failed: {result.extraction.errors}")
            return

        chunks = result.chunks
        logger.info(f"[{task_id}] Extracted {len(chunks)} chunks")

        # Step 2: Embed (CPU-bound — offload to thread)
        texts = [c["text"] for c in chunks]
        embeddings = await loop.run_in_executor(None, generate_embeddings, texts)

        # Step 3: Store in Qdrant
        # add_documents is now sync — offload to thread so we don't block the loop
        vector_provider = VectorSearchProvider(collection_name="pdf_documents")
        await loop.run_in_executor(
            None, vector_provider.add_documents, chunks, embeddings
        )

        logger.info(f"[{task_id}] Successfully stored {len(chunks)} chunks in Qdrant")

    except Exception as e:
        logger.error(f"[{task_id}] Error processing PDF: {str(e)}", exc_info=True)