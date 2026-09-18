"""FastAPI application entry point for QueryNest."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import ingest, query

app = FastAPI(
    title="QueryNest API",
    description="A private document Q&A and RAG platform for uploading files and querying local knowledge bases.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])


@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "service": "QueryNest",
        "version": "1.0.0",
        "message": "QueryNest API is running",
    }


@app.get("/health", tags=["health"])
def health():
    from api.state import rag_state
    return {
        "status": "ok",
        "service": "QueryNest",
        "document_count": len(rag_state.store),
    }


@app.get("/documents", tags=["ingest"])
def list_documents():
    """List all ingested document sources."""
    from api.state import rag_state
    sources = list({c.metadata.get("source", "unknown") for c in rag_state.store._chunks})
    return {"count": len(sources), "sources": sorted(sources)}
