# QueryNest

QueryNest is a local-first document question-answering application built around a custom Retrieval-Augmented Generation (RAG) pipeline. Upload TXT, Markdown, or PDF files, index their content, and ask questions with answers grounded in the retrieved source material.

The project combines a FastAPI backend, a lightweight browser client, and independently implemented ingestion, chunking, embedding, retrieval, and generation components.

## Highlights

- Upload and index `.txt`, `.md`, and `.pdf` documents
- Track ingestion progress from upload through vector indexing
- Split documents into overlapping searchable chunks
- Generate local embeddings with Sentence Transformers
- Retrieve relevant context with cosine similarity search
- Generate answers with Ollama or OpenAI-compatible backends
- Return source citations with relevance scores and snippets
- Use the responsive frontend without a JavaScript build step

## Architecture

```text
Browser client
    |
    | upload document / ask question
    v
FastAPI application
    |
    +-- Ingestion route
    |     +-- validate upload
    |     +-- extract document text
    |     +-- create overlapping chunks
    |     +-- generate embeddings
    |     +-- store vectors in memory
    |
    +-- Query route
          +-- embed the question
          +-- retrieve top-k chunks
          +-- build grounded prompt
          +-- generate answer
          +-- return citations
```

The runtime state is intentionally local and in memory. Restarting the API clears the current document index.

## Repository structure

```text
rag_from_scratch/
├── api/
│   ├── main.py                 # FastAPI application and health endpoints
│   ├── models.py               # Request and response schemas
│   ├── state.py                # Shared RAG service state
│   └── routes/
│       ├── ingest.py           # Upload and indexing workflow
│       └── query.py            # Retrieval and answer generation
├── frontend/
│   ├── index.html              # Application shell
│   ├── app.js                  # Upload, polling, and chat behavior
│   ├── style.css               # Responsive product interface
│   └── favicon.svg             # Browser icon
├── rag/
│   ├── loader.py               # TXT, Markdown, and PDF loading
│   ├── chunker.py              # Chunking strategies
│   ├── embedder.py             # Embedding backends
│   ├── vector_store.py         # In-memory vector search
│   ├── retriever.py            # Top-k retrieval
│   ├── prompt_builder.py       # Grounded prompt construction
│   ├── generator.py            # LLM integrations
│   └── evaluator.py            # Retrieval evaluation helpers
├── tests/                      # Unit and API tests
├── .env.example                # Safe configuration template
├── requirements.txt            # Python dependencies
└── README.md
```

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/Bhavyasri212/Rag_from_scratch.git
cd Rag_from_scratch
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the application

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS or Linux:

```bash
cp .env.example .env
```

The default configuration uses local Sentence Transformers embeddings and Ollama generation. Install Ollama separately, then make a model available:

```bash
ollama pull llama3
```

### 5. Start the backend

```bash
python -m uvicorn api.main:app --reload
```

The API is available at `http://localhost:8000`.

### 6. Start the frontend

In a second terminal:

```bash
python -m http.server 5500 --directory frontend
```

Open `http://localhost:5500` in a browser.

## API reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Service information |
| `GET` | `/health` | API and document-index health |
| `GET` | `/documents` | List indexed document sources |
| `POST` | `/ingest/` | Queue a document for indexing |
| `GET` | `/ingest/status/{job_id}` | Read indexing progress |
| `POST` | `/query/` | Ask a question against indexed content |
| `GET` | `/docs` | Interactive Swagger documentation |
| `GET` | `/redoc` | ReDoc API documentation |

Example query request:

```json
{
  "question": "What are the main ideas in this document?",
  "k": 5
}
```

The query response includes the generated answer and source citations containing the source name, page number, similarity score, and retrieved snippet.

## Configuration

Copy `.env.example` to `.env` and adjust the values for your selected providers.

| Variable | Description | Default |
| --- | --- | --- |
| `EMBEDDING_BACKEND` | `local` or `openai` | `local` |
| `LOCAL_EMBEDDING_MODEL` | Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `LLM_BACKEND` | `ollama` or `openai` | `ollama` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama generation model | `llama3` |
| `OLLAMA_KEEP_ALIVE` | Ollama model lifetime | `10m` |
| `OLLAMA_NUM_PREDICT` | Maximum generated tokens | `256` |
| `CHUNK_SIZE` | Chunk size in characters | `500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `TOP_K` | Default retrieval depth | `5` |
| `OPENAI_API_KEY` | Required for OpenAI backends | Not set |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `OPENAI_LLM_MODEL` | OpenAI chat model | `gpt-4o-mini` |

Never commit `.env` or API keys. Use `.env.example` as the public configuration template.

## Testing

Run the complete test suite from the repository root:

```bash
pytest -q
```

The tests cover document loading, text chunking, vector-store behavior, and basic API health/document endpoints.

