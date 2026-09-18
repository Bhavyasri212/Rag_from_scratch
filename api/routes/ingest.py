"""POST /ingest — upload a file, chunk it, embed it, add to the vector store."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from api.models import IngestJobResponse, IngestStatusResponse
from api.state import rag_state
from rag.chunker import chunk_documents
from rag.loader import load_document

router = APIRouter()

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


@router.post("/", response_model=IngestJobResponse)
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Queue an upload and return immediately with a progress job ID."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}")

    # Write the upload to a temp file so our loader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    job_id = rag_state.create_job(file.filename)
    background_tasks.add_task(process_upload, job_id, file.filename, tmp_path)
    return IngestJobResponse(job_id=job_id, filename=file.filename)


def process_upload(job_id: str, filename: str, tmp_path: Path):
    """Run the ingestion pipeline while updating the job visible to the UI."""
    try:
        rag_state.update_job(job_id, status="processing", stage="reading", progress=18, message="Extracting document text")
        docs = load_document(tmp_path)
        for doc in docs:
            doc["metadata"]["source"] = filename

        rag_state.update_job(job_id, stage="chunking", progress=38, message="Splitting text into searchable chunks")
        chunks = chunk_documents(docs, strategy="overlap")

        rag_state.update_job(job_id, stage="embedding", progress=62, message=f"Generating embeddings for {len(chunks)} chunks")
        vectors = rag_state.embedder.embed_chunks(chunks)

        rag_state.update_job(job_id, stage="indexing", progress=86, message="Adding vectors to the search index")
        rag_state.store.add_batch(chunks, vectors)
        rag_state.update_job(
            job_id,
            status="complete",
            stage="ready",
            progress=100,
            message="Source is ready for questions",
            chunks_added=len(chunks),
            total_chunks=len(rag_state.store),
        )
    except Exception as exc:
        rag_state.update_job(job_id, status="error", stage="error", progress=100, message="Indexing failed", error=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/status/{job_id}", response_model=IngestStatusResponse)
def ingest_status(job_id: str):
    """Return the current stage of an ingestion job."""
    job = rag_state.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job
