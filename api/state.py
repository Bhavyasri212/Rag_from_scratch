"""Shared application state — single in-memory VectorStore and Embedder for all requests."""

from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from threading import Lock
from uuid import uuid4


class RAGState:
    """Holds all runtime components as a singleton."""
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.retriever = Retriever(self.store, self.embedder)
        self.generator = Generator()
        self.jobs = {}
        self.jobs_lock = Lock()

    def create_job(self, filename: str) -> str:
        job_id = str(uuid4())
        with self.jobs_lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "filename": filename,
                "status": "queued",
                "stage": "upload",
                "progress": 5,
                "message": "Upload received",
                "chunks_added": 0,
                "total_chunks": 0,
                "error": None,
            }
        return job_id

    def update_job(self, job_id: str, **updates):
        with self.jobs_lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)

    def get_job(self, job_id: str):
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None


rag_state = RAGState()
