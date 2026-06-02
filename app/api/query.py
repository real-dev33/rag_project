# app/api/query.py
import time
import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.embeddings import generate_embedding
from app.services.llm_service import generate_answer, LLMProvider
from app.tools.vector_search import VectorSearchProvider

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    provider: LLMProvider = LLMProvider.OPENROUTER   # switch to "groq" in request body


class ChunkResult(BaseModel):
    text: str
    page: Optional[int] = None
    score: float
    chunk_type: Optional[str] = None
    metadata: Dict[str, Any] = {}


class Source(BaseModel):
    text: str
    page: Optional[int] = None
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str                      # LLM-generated natural language answer
    sources: List[Source]            # chunks the LLM cited
    results: List[ChunkResult]       # all retrieved chunks (for debugging)
    total_results: int
    latency_ms: int


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG system",
    description=(
        "Embeds the question, retrieves relevant chunks from Qdrant, "
        "then feeds them to an LLM to produce a grounded natural language answer."
    ),
)
def query(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline:
      1. Embed question → vector
      2. Search Qdrant → top-k chunks
      3. Feed chunks + question to LLM → grounded answer with citations
    """
    start = time.monotonic()

    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="question must not be empty",
        )

    # Step 1: embed the question
    try:
        query_vector = generate_embedding(request.question)
    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate query embedding",
        )

    # Step 2: retrieve chunks from Qdrant
    try:
        provider = VectorSearchProvider(collection_name="pdf_documents")
        raw_results = provider.search(
            query_vector=query_vector,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector search failed",
        )

    # Step 3: generate LLM answer from retrieved chunks
    try:
        llm_result = generate_answer(
            question=request.question,
            chunks=raw_results,
            provider=request.provider,
        )
    except ValueError as e:
        # Missing API key — surface clearly rather than 500
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM answer generation failed",
        )

    # Assemble response
    results = [
        ChunkResult(
            text=r["text"],
            page=r.get("page"),
            score=round(r["score"], 4),
            chunk_type=r.get("chunk_type"),
            metadata=r.get("metadata", {}),
        )
        for r in raw_results
    ]

    sources = [
        Source(
            text=s["text"],
            page=s.get("page"),
            score=round(s["score"], 4),
        )
        for s in llm_result["sources"]
    ]

    latency_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        f"[{request.provider.value}] '{request.question[:60]}' → "
        f"{len(results)} chunks, {len(sources)} cited, {latency_ms}ms"
    )

    return QueryResponse(
        question=request.question,
        answer=llm_result["answer"],
        sources=sources,
        results=results,
        total_results=len(results),
        latency_ms=latency_ms,
    )