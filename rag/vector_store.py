"""Custom vector store — stores embeddings and finds similar ones using cosine similarity."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunker import Chunk


@dataclass
class ScoredChunk:
    """A retrieved chunk paired with its similarity score."""
    chunk: Chunk
    score: float  # cosine similarity in [0, 1]


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    Formula:  cos(θ) = (a · b) / (||a|| * ||b||)

    Returns 1.0 for identical direction, 0.0 for orthogonal, -1.0 for opposite.
    We normalize first (dot product of unit vectors = cosine similarity).
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a / a_norm, b / b_norm))


# ── Vector Store ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    In-memory vector store backed by two parallel lists:
      - self._vectors : list of numpy arrays (one per chunk)
      - self._chunks  : list of Chunk objects (same order)

    Search is O(N * D) brute-force — fine for small corpora.
    For large-scale use, swap in FAISS or Annoy.
    """

    def __init__(self):
        self._vectors: list[np.ndarray] = []  # raw embeddings
        self._chunks: list[Chunk] = []         # corresponding chunks

    def add(self, chunk: Chunk, vector: np.ndarray) -> None:
        """Store a chunk and its embedding vector."""
        self._vectors.append(np.array(vector, dtype=np.float32))
        self._chunks.append(chunk)

    def add_batch(self, chunks: list[Chunk], vectors: list[np.ndarray]) -> None:
        """Add multiple chunks and their vectors at once."""
        for chunk, vector in zip(chunks, vectors):
            self.add(chunk, vector)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[ScoredChunk]:
        """Return the top-k chunks most similar to the query vector."""
        if not self._vectors:
            return []

        # Compute cosine similarity against every stored vector
        scores = [cosine_similarity(query_vector, v) for v in self._vectors]

        # Pair each score with its index, sort descending, take top-k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        return [ScoredChunk(chunk=self._chunks[i], score=score) for i, score in ranked]

    def __len__(self) -> int:
        return len(self._chunks)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the store to a JSON file (vectors as lists, metadata preserved)."""
        path = Path(path)
        data = {
            "vectors": [v.tolist() for v in self._vectors],
            "chunks": [
                {"content": c.content, "metadata": c.metadata}
                for c in self._chunks
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved {len(self)} chunks → {path}")

    def load(self, path: str | Path) -> None:
        """Load a previously saved store from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self._vectors = [np.array(v, dtype=np.float32) for v in data["vectors"]]
        self._chunks = [
            Chunk(content=c["content"], metadata=c["metadata"])
            for c in data["chunks"]
        ]
        print(f"Loaded {len(self)} chunks ← {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rag.loader import load_document
    from rag.chunker import chunk_documents
    from rag.embedder import Embedder

    path = sys.argv[1] if len(sys.argv) > 1 else "docs/artificial_intelligence.txt"
    query = sys.argv[2] if len(sys.argv) > 2 else "What is machine learning?"

    docs = load_document(path)
    chunks = chunk_documents(docs, strategy="overlap", chunk_size=300, overlap=50)

    embedder = Embedder()
    store = VectorStore()
    store.add_batch(chunks, embedder.embed_chunks(chunks))

    print(f"Store: {len(store)} chunks\n")
    results = store.search(embedder.embed(query), k=3)
    print(f"Query: '{query}'\n")
    for r in results:
        print(f"  score={r.score:.4f} | {r.chunk.content[:120]}...\n")
