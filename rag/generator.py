"""Generator — sends the built prompt to Ollama (or OpenAI) and returns the answer."""

from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

from rag.vector_store import ScoredChunk
from rag.prompt_builder import build_prompt, format_sources

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
OPENAI_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")


@dataclass
class RAGResponse:
    """Holds the generated answer and its citation sources."""
    answer: str
    sources: list[dict]
    question: str


class Generator:
    """Sends a prompt to an LLM and returns a structured RAGResponse."""

    def __init__(self, backend: str = LLM_BACKEND):
        self.backend = backend

    def generate(self, question: str, results: list[ScoredChunk]) -> RAGResponse:
        """Build the prompt, call the LLM, and return the answer with sources."""
        messages = build_prompt(question, results)
        sources = format_sources(results)

        if self.backend == "ollama":
            answer = self._call_ollama(messages)
        elif self.backend == "openai":
            answer = self._call_openai(messages)
        else:
            raise ValueError(f"Unknown LLM backend '{self.backend}'. Use 'ollama' or 'openai'.")

        return RAGResponse(answer=answer, sources=sources, question=question)

    def _call_ollama(self, messages: list[dict]) -> str:
        """Send messages to a local Ollama server."""
        import ollama
        client = ollama.Client(host=OLLAMA_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "num_predict": OLLAMA_NUM_PREDICT,
                "temperature": 0.2,
            },
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        return response["message"]["content"].strip()

    def _call_openai(self, messages: list[dict]) -> str:
        """Send messages to the OpenAI chat completions API."""
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,  # low temperature for factual grounded answers
        )
        return response.choices[0].message.content.strip()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rag.loader import load_document
    from rag.chunker import chunk_documents
    from rag.embedder import Embedder
    from rag.vector_store import VectorStore
    from rag.retriever import Retriever

    path = sys.argv[1] if len(sys.argv) > 1 else "docs/artificial_intelligence.txt"
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What is RAG?"

    docs = load_document(path)
    chunks = chunk_documents(docs, strategy="overlap", chunk_size=400, overlap=60)
    embedder = Embedder()
    store = VectorStore()
    store.add_batch(chunks, embedder.embed_chunks(chunks))
    retriever = Retriever(store, embedder)
    results = retriever.retrieve(query, k=4)

    gen = Generator()
    resp = gen.generate(query, results)

    print(f"Q: {resp.question}\n{'─'*60}")
    print(f"A: {resp.answer}\n")
    print("Sources:")
    for s in resp.sources:
        print(f"  [{s['score']:.4f}] {s['source']}  — {s['snippet'][:80]}...")
