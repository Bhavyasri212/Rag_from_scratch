"""POST /query — answer a question using the current vector store."""

from fastapi import APIRouter, HTTPException

from api.models import QueryRequest, QueryResponse, SourceCitation
from api.state import rag_state

router = APIRouter()


@router.post("/", response_model=QueryResponse)
def answer_question(body: QueryRequest):
    """Retrieve relevant chunks and generate a grounded answer."""
    if len(rag_state.store) == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Upload a file first.")

    results = rag_state.retriever.retrieve(body.question, k=body.k)
    response = rag_state.generator.generate(body.question, results)

    sources = [
        SourceCitation(
            source=s["source"],
            page=s["page"],
            score=s["score"],
            snippet=s["snippet"],
        )
        for s in response.sources
    ]

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=sources,
    )
