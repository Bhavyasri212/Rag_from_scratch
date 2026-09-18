"""Embedding generation — converts text into dense vectors using local or OpenAI models."""

from __future__ import annotations
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
LOCAL_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


class Embedder:
    """Wraps a local (sentence-transformers) or OpenAI embedding model."""

    def __init__(self, backend: str = BACKEND):
        self.backend = backend
        self._model = None  # lazy-loaded on first call

    def _load_model(self):
        """Load the model on first use to avoid slow startup."""
        if self._model is not None:
            return
        if self.backend == "local":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(LOCAL_MODEL)
        elif self.backend == "openai":
            from openai import OpenAI
            self._model = OpenAI()
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Use 'local' or 'openai'.")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string → 1-D numpy array."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of strings → list of 1-D numpy arrays."""
        self._load_model()

        if self.backend == "local":
            # sentence-transformers returns a 2-D numpy array (batch_size, dim)
            vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return [v for v in vectors]

        elif self.backend == "openai":
            response = self._model.embeddings.create(model=OPENAI_MODEL, input=texts)
            return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    def embed_chunks(self, chunks) -> list[np.ndarray]:
        """Convenience: embed a list of Chunk objects by their content."""
        texts = [c.content for c in chunks]
        return self.embed_batch(texts)

    @property
    def dimension(self) -> int:
        """Return the embedding vector size (useful for setting up vector store)."""
        test_vec = self.embed("test")
        return len(test_vec)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rag.loader import load_document
    from rag.chunker import chunk_documents

    path = sys.argv[1] if len(sys.argv) > 1 else "docs/artificial_intelligence.txt"
    docs = load_document(path)
    chunks = chunk_documents(docs, strategy="overlap", chunk_size=300, overlap=50)[:3]

    embedder = Embedder()
    print(f"Backend : {embedder.backend}")
    print(f"Dim     : {embedder.dimension}")
    print(f"Chunks  : {len(chunks)} (showing embeddings for first 3)\n")

    for i, (chunk, vec) in enumerate(zip(chunks, embedder.embed_chunks(chunks))):
        print(f"[chunk {i}] shape={vec.shape}  norm={np.linalg.norm(vec):.4f}  preview: {chunk.content[:80]}...")
