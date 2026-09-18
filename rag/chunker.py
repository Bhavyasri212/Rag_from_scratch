"""Text chunking — splits documents into smaller pieces for embedding."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A piece of text with its origin metadata."""
    content: str
    metadata: dict = field(default_factory=dict)  # inherits parent doc metadata + chunk fields


# ── Fixed-size chunker ────────────────────────────────────────────────────────

def fixed_size_chunks(document: dict, chunk_size: int = 500) -> list[Chunk]:
    """Split text every chunk_size characters — simplest possible strategy."""
    text = document["content"]
    meta = document["metadata"]
    chunks = []
    for i, start in enumerate(range(0, len(text), chunk_size)):
        piece = text[start: start + chunk_size]
        if piece.strip():
            chunks.append(Chunk(
                content=piece,
                metadata={**meta, "chunk_index": i, "char_start": start, "strategy": "fixed"},
            ))
    return chunks


# ── Overlap chunker ───────────────────────────────────────────────────────────

def overlap_chunks(document: dict, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Fixed-size chunks with a sliding window overlap to preserve context at boundaries."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document["content"]
    meta = document["metadata"]
    step = chunk_size - overlap  # how far we advance each iteration
    chunks = []
    idx = 0
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start: start + chunk_size]
        if piece.strip():
            chunks.append(Chunk(
                content=piece,
                metadata={**meta, "chunk_index": i, "char_start": start, "strategy": "overlap"},
            ))
        if start + chunk_size >= len(text):
            break
    return chunks


# ── Paragraph chunker ─────────────────────────────────────────────────────────

def paragraph_chunks(document: dict, max_chars: int = 1000) -> list[Chunk]:
    """Split on blank lines; merge short paragraphs to stay under max_chars."""
    text = document["content"]
    meta = document["metadata"]

    # Split on one or more blank lines
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    chunks = []
    current, current_start, char_cursor = [], 0, 0

    for para in paragraphs:
        # If adding this paragraph would exceed the limit, flush current buffer
        combined_len = sum(len(p) for p in current) + len(para)
        if current and combined_len > max_chars:
            content = "\n\n".join(current)
            chunks.append(Chunk(
                content=content,
                metadata={**meta, "chunk_index": len(chunks), "char_start": current_start, "strategy": "paragraph"},
            ))
            current_start = char_cursor
            current = []

        current.append(para)
        char_cursor += len(para) + 2  # +2 for the "\n\n" separator

    # Flush remaining paragraphs
    if current:
        chunks.append(Chunk(
            content="\n\n".join(current),
            metadata={**meta, "chunk_index": len(chunks), "char_start": current_start, "strategy": "paragraph"},
        ))

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

STRATEGIES = {
    "fixed": fixed_size_chunks,
    "overlap": overlap_chunks,
    "paragraph": paragraph_chunks,
}


def chunk_document(document: dict, strategy: str = "overlap", **kwargs) -> list[Chunk]:
    """Chunk a single document using the named strategy."""
    fn = STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES)}")
    return fn(document, **kwargs)


def chunk_documents(documents: list[dict], strategy: str = "overlap", **kwargs) -> list[Chunk]:
    """Chunk a list of documents and flatten all chunks into one list."""
    result = []
    for doc in documents:
        result.extend(chunk_document(doc, strategy=strategy, **kwargs))
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rag.loader import load_document

    if len(sys.argv) < 2:
        print("Usage: python -m rag.chunker <file> [strategy] [chunk_size]")
        sys.exit(1)

    path = sys.argv[1]
    strategy = sys.argv[2] if len(sys.argv) > 2 else "overlap"
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    docs = load_document(path)
    chunks = chunk_documents(docs, strategy=strategy, chunk_size=size)

    print(f"Strategy: {strategy}  |  {len(chunks)} chunks from {path}\n")
    for c in chunks[:3]:  # preview first 3
        print(f"[chunk {c.metadata['chunk_index']}] chars={len(c.content)}")
        print(c.content[:200], "...\n")
