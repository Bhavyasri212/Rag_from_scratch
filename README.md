# QueryNest

QueryNest is a local-first, document-aware AI assistant that lets users upload text-based source files, index them into a custom retrieval pipeline, and ask grounded questions about the content.

This project demonstrates a production-style Retrieval-Augmented Generation (RAG) workflow built from scratch using Python, FastAPI, and a lightweight frontend interface.

---

## Why this project matters

This project was built to show:

- document ingestion and preprocessing
- chunking and retrieval logic
- embedding-based semantic search
- prompt construction with context grounding
- a basic AI-powered Q&A workflow over local files

It is a strong portfolio project because it combines core AI engineering concepts with a usable product interface.

---

## Architecture overview

```text
User uploads file
       │
       ▼
Frontend UI (HTML + CSS + JS)
       │
       ▼
FastAPI backend
       │
       ├── file validation and upload handling
       ├── ingestion job tracking
       ├── chunking and indexing pipeline
       └── query endpoint with retrieval + generation
       │
       ▼
Custom RAG components
- loader.py  -> read and normalize input files
- chunker.py -> split text into overlapping chunks
- embedder.py -> generate vector representations
- vector_store.py -> store and search embeddings
- retriever.py -> return top-k relevant chunks
- generator.py -> produce grounded answers

Final result: answer + citations + source references
```

---

## Project structure

```text
rag_from_scratch/
├── api/
│   ├── main.py
│   ├── models.py
│   ├── state.py
│   └── routes/
│       ├── ingest.py
│       └── query.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   ├── favicon.svg
│   └── assets/
├── rag/
│   ├── chunker.py
│   ├── embedder.py
│   ├── evaluator.py
│   ├── generator.py
│   ├── loader.py
│   ├── prompt_builder.py
│   ├── retriever.py
│   └── vector_store.py
├── tests/
│   ├── test_loader.py
│   └── test_api_health.py
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md
```

---

## Features

- upload local documents in TXT, MD, or PDF format
- parse and normalize content
- split text into smaller knowledge chunks
- embed and store document chunks in memory
- retrieve the most relevant chunks for a question
- return grounded answers based on matched context
- show document metadata and source references in the UI
- provide a responsive interface for local document Q&A

---

## Local setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd rag_from_scratch

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the example environment file
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux

# 5. Start the backend
uvicorn api.main:app --reload
```

Then open the frontend locally:

```bash
python -m http.server 5500 --directory frontend
```

Visit:

- API: http://localhost:8000/docs
- Frontend: http://localhost:5500

---

## Environment variables

Use the values in `.env.example` as a template.

| Variable                | Purpose                             |
| ----------------------- | ----------------------------------- |
| `EMBEDDING_BACKEND`     | Select local or OpenAI embeddings   |
| `LOCAL_EMBEDDING_MODEL` | Default sentence-transformers model |
| `LLM_BACKEND`           | Use Ollama or OpenAI generation     |
| `OLLAMA_MODEL`          | Local model name for Ollama         |
| `CHUNK_SIZE`            | Number of characters per chunk      |
| `CHUNK_OVERLAP`         | Overlap between adjacent chunks     |
| `TOP_K`                 | Number of chunks to retrieve        |

---

## Testing

```bash
pytest
```

The project includes loader and API health checks to validate core functionality.

---

## Deployment notes

This project is designed as a small full-stack AI application. The frontend can be served as static files, while the FastAPI backend can run on Render, Railway, or another Python-capable host. Store environment variables in the hosting provider rather than committing them.

Because the current app stores indexed data in memory, deployment should be paired with persistent storage if you want a production-grade multi-user system.

---

## Resume-ready summary

QueryNest is a full-stack Retrieval-Augmented Generation application for document-based Q&A. It includes file ingestion, chunking, embeddings, semantic retrieval, and a polished frontend interface for querying local knowledge sources.

This project demonstrates practical AI engineering skills in backend API design, vector retrieval, prompt grounding, and product-level UX for a document assistant.
