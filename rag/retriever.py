"""Retriever — embeds a question and finds the most relevant chunks."""

from __future__ import annotations
import os
from dotenv import load_dotenv

from rag.embedder import Embedder
from rag.vector_store import VectorStore, ScoredChunk

load_dotenv()
TOP_K = int(os.getenv("TOP_K", 5))


class Retriever:
    """Combines an Embedder and a VectorStore to answer: 'what chunks match this query?'"""

    def __init__(self, store: VectorStore, embedder: Embedder | None = None):
        self.store = store
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, k: int = TOP_K) -> list[ScoredChunk]:
        """Embed the query then return the top-k most similar chunks."""
        query_vector = self.embedder.embed(query)
        results = self.store.search(query_vector, k=k)
        return results

    def retrieve_above(self, query: str, threshold: float = 0.4, k: int = TOP_K) -> list[ScoredChunk]:
        """Return only chunks whose similarity score exceeds a threshold."""
        return [r for r in self.retrieve(query, k=k) if r.score >= threshold]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rag.loader import load_document
    from rag.chunker import chunk_documents

    path = sys.argv[1] if len(sys.argv) > 1 else "docs/artificial_intelligence.txt"
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What is deep learning?"

    docs = load_document(path)
    chunks = chunk_documents(docs, strategy="overlap", chunk_size=400, overlap=60)

    embedder = Embedder()
    store = VectorStore()
    store.add_batch(chunks, embedder.embed_chunks(chunks))

    retriever = Retriever(store, embedder)
    results = retriever.retrieve(query, k=3)

    print(f"Query: '{query}'\n{'─'*60}")
    for r in results:
        src = r.chunk.metadata.get("source", "?")
        print(f"score={r.score:.4f} | {src}\n{r.chunk.content[:250]}...\n")
