"""Document ingestion — reads .txt, .md, and .pdf into a unified dict format."""

from __future__ import annotations
import re
import sys
from pathlib import Path

# Each document is a plain dict: {"content": str, "metadata": dict}
Document = dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # unify line endings
    text = re.sub(r"\n{3,}", "\n\n", text)                   # collapse blank lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))  # strip trailing spaces
    lines = [re.sub(r"[ \t]+", " ", line).lstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _metadata(source: str | Path, file_type: str, page: int, char_count: int) -> dict:
    """Build the standard metadata dict."""
    return {"source": str(source), "file_type": file_type, "page": page, "char_count": char_count}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_txt(path: str | Path) -> list[Document]:
    """Load a .txt file and return it as a single Document."""
    path = Path(path)
    content = _clean_text(path.read_text(encoding="utf-8", errors="replace"))
    return [{"content": content, "metadata": _metadata(path, "txt", 0, len(content))}]


def load_md(path: str | Path) -> list[Document]:
    """Load a .md file — Markdown syntax is kept because LLMs understand it."""
    path = Path(path)
    content = _clean_text(path.read_text(encoding="utf-8", errors="replace"))
    return [{"content": content, "metadata": _metadata(path, "md", 0, len(content))}]


def load_pdf(path: str | Path) -> list[Document]:
    """Load a .pdf file and return one Document per page."""
    path = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf: pip install pypdf") from exc

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        content = _clean_text(page.extract_text() or "")
        if content:  # skip blank or image-only pages
            docs.append({"content": content, "metadata": _metadata(path, "pdf", page_num, len(content))})
    return docs


# ── Dispatcher ────────────────────────────────────────────────────────────────

_LOADERS = {".txt": load_txt, ".md": load_md, ".pdf": load_pdf}


def load_document(path: str | Path) -> list[Document]:
    """Detect file type from extension and dispatch to the right loader."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported type '{path.suffix}'. Supported: {list(_LOADERS)}")
    return loader(path)


def load_documents(paths: list[str | Path]) -> list[Document]:
    """Load multiple files and flatten all Documents into one list."""
    docs: list[Document] = []
    for p in paths:
        docs.extend(load_document(p))
    return docs


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Usage: python -m rag.loader docs/artificial_intelligence.txt
    if len(sys.argv) < 2:
        print("Usage: python -m rag.loader <file_path>")
        sys.exit(1)

    for doc in load_document(sys.argv[1]):
        m = doc["metadata"]
        print(f"[{m['file_type']}] {m['source']}  page={m['page']}  chars={m['char_count']}")
        print(doc["content"][:300], "...\n")
