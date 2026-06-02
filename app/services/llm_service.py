# app/services/llm_service.py
import logging
import re
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"


# ── Context assembly ───────────────────────────────────────────────────────────

def build_context(chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Assemble retrieved chunks into a single numbered context string.

    Chunks are already ranked by score (highest first) from Qdrant.
    We truncate at max_chars to stay within LLM context limits.
    """
    parts = []
    total = 0

    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("page")
        page_str = f"Page {page}" if page else "Page unknown"
        header = f"[Source {i} | {page_str}]"
        body = chunk["text"].strip()
        block = f"{header}\n{body}"

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining] + "…")
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """Build the RAG prompt sent to the LLM."""
    return f"""You are a precise document assistant. Answer the question using ONLY the provided context.

Rules:
- Cite sources using their [Source N] label, e.g. "According to [Source 2]..."
- Include page numbers when available, e.g. "[Source 1, Page 4]"
- If the answer is not in the context, say: "I couldn't find information about this in the provided documents."
- Do not make up information or draw on outside knowledge.
- Be concise but complete.

Context:
{context}

Question: {question}

Answer:"""


# ── Provider implementations ───────────────────────────────────────────────────

def _call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini via new google-genai SDK."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


def _call_groq(prompt: str, api_key: str) -> str:
    """Call Groq (llama-3.1-8b-instant) via groq SDK."""
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq not installed. Run: pip install groq")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _call_openrouter(prompt: str, api_key: str) -> str:
    """
    Call OpenRouter using the OpenAI-compatible API.

    Model: openrouter/free (auto-selects best free model, currently gpt-3.5-turbo-0613)
    Free tier, no rate limit issues, works with Gmail accounts.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai not installed. Run: pip install openai")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ── Public interface ───────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    provider: LLMProvider = LLMProvider.OPENROUTER,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a natural language answer from retrieved chunks.

    Args:
        question: The user's question
        chunks: Retrieved chunks from vector search (ranked by score)
        provider: Which LLM to use — defaults to openrouter
        api_key: API key — resolved from settings if not passed directly

    Returns:
        Dict with 'answer', 'context_used', 'sources' keys
    """
    if not chunks:
        return {
            "answer": "No relevant content was found in the documents to answer this question.",
            "context_used": "",
            "sources": [],
        }

    # Build context and prompt
    context = build_context(chunks)
    prompt = build_prompt(question, context)

    # Resolve API key from settings if not passed directly
    if api_key is None:
        from app.core.config import settings
        if provider == LLMProvider.GEMINI:
            api_key = settings.gemini_api_key
        elif provider == LLMProvider.GROQ:
            api_key = settings.groq_api_key
        elif provider == LLMProvider.OPENROUTER:
            api_key = settings.openrouter_api_key

        if not api_key:
            key_name = {
                LLMProvider.GEMINI: "GEMINI_API_KEY",
                LLMProvider.GROQ: "GROQ_API_KEY",
                LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
            }[provider]
            raise ValueError(f"{key_name} not set — add it to your .env file")

    logger.info(f"[{provider.value}] Generating answer for: '{question[:60]}'")

    if provider == LLMProvider.GEMINI:
        answer = _call_gemini(prompt, api_key)
    elif provider == LLMProvider.GROQ:
        answer = _call_groq(prompt, api_key)
    elif provider == LLMProvider.OPENROUTER:
        answer = _call_openrouter(prompt, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Extract which sources were cited in the answer
    cited = sorted(set(int(n) for n in re.findall(r'\[Source (\d+)', answer)))
    sources = [chunks[i - 1] for i in cited if 0 < i <= len(chunks)]

    logger.info(f"Answer generated ({provider.value}), cited {len(cited)} sources")

    return {
        "answer": answer,
        "context_used": context,
        "sources": sources,
    }