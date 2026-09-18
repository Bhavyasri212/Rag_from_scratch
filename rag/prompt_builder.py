"""Prompt builder — assembles the final prompt from retrieved chunks and the user's question."""

from __future__ import annotations
from rag.vector_store import ScoredChunk

# System instruction sent before the context
SYSTEM_PROMPT = (
    "You are a knowledgeable document assistant. Answer the user's question using ONLY "
    "the provided context. Give a useful, well-organized answer rather than a one-line "
    "summary: start with the direct answer, then explain the important details from the "
    "context. Use short paragraphs or bullet points when they improve readability. "
    "For comparison or process questions, use a numbered list. Do not add inline source "
    "markers, bracketed citations, or document filenames to the answer. Do not mention "
    "the retrieval process, do not say \"according to the context\", and do not invent information. "
    "If the answer is not supported by the context, say \"I don't know based on the "
    "provided documents.\""
)


def build_prompt(question: str, results: list[ScoredChunk]) -> list[dict]:
    """
    Build an OpenAI-style messages list (works with Ollama too).

    Structure:
        [system]  → instructions + grounding rule
        [user]    → context blocks + question
    """
    context_blocks = []
    for r in results:
        source = r.chunk.metadata.get("source", "unknown")
        page = r.chunk.metadata.get("page", 0)
        label = f"Source document: {source}" + (f", page {page}" if page else "")
        context_blocks.append(f"{label}\n{r.chunk.content}")

    context_text = "\n\n".join(context_blocks)

    user_content = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def format_sources(results: list[ScoredChunk]) -> list[dict]:
    """Return a clean list of citation dicts for attaching to the response."""
    sources_by_document = {}
    for r in results:
        source = r.chunk.metadata.get("source", "unknown")
        citation = {
            "source": source,
            "page": r.chunk.metadata.get("page", 0),
            "score": round(r.score, 4),
            "snippet": r.chunk.content[:150],
        }
        existing = sources_by_document.get(source)
        if existing is None or citation["score"] > existing["score"]:
            sources_by_document[source] = citation
    return list(sources_by_document.values())
