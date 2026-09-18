"""Pydantic schemas for request/response bodies."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class SourceCitation(BaseModel):
    source: str
    page: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceCitation]


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


class IngestJobResponse(BaseModel):
    job_id: str
    filename: str


class IngestStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    stage: str
    progress: int
    message: str
    chunks_added: int = 0
    total_chunks: int = 0
    error: str | None = None
