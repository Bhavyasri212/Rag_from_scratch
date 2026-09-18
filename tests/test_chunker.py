"""Unit tests for rag/chunker.py"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.chunker import fixed_size_chunks, overlap_chunks, paragraph_chunks, chunk_document, chunk_documents

# Shared dummy document
DOC = {"content": "ABCDE" * 100, "metadata": {"source": "test.txt", "file_type": "txt", "page": 0}}
PARA_DOC = {
    "content": "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.",
    "metadata": {"source": "test.txt", "file_type": "txt", "page": 0},
}


class TestFixedSizeChunks:
    def test_splits_into_correct_number_of_chunks(self):
        chunks = fixed_size_chunks(DOC, chunk_size=100)
        assert len(chunks) == 5  # 500 chars / 100 = 5

    def test_each_chunk_has_correct_size(self):
        chunks = fixed_size_chunks(DOC, chunk_size=100)
        for c in chunks[:-1]:  # last chunk may be shorter
            assert len(c.content) == 100

    def test_metadata_strategy_is_fixed(self):
        chunks = fixed_size_chunks(DOC, chunk_size=100)
        assert all(c.metadata["strategy"] == "fixed" for c in chunks)

    def test_chunk_index_increments(self):
        chunks = fixed_size_chunks(DOC, chunk_size=100)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(5))

    def test_parent_metadata_preserved(self):
        chunks = fixed_size_chunks(DOC, chunk_size=100)
        assert all(c.metadata["source"] == "test.txt" for c in chunks)


class TestOverlapChunks:
    def test_produces_more_chunks_than_fixed(self):
        fixed = fixed_size_chunks(DOC, chunk_size=100)
        overlapped = overlap_chunks(DOC, chunk_size=100, overlap=20)
        assert len(overlapped) > len(fixed)

    def test_consecutive_chunks_share_content(self):
        chunks = overlap_chunks(DOC, chunk_size=100, overlap=20)
        # End of chunk N should appear at start of chunk N+1
        end_of_first = chunks[0].content[-20:]
        start_of_second = chunks[1].content[:20]
        assert end_of_first == start_of_second

    def test_raises_if_overlap_gte_chunk_size(self):
        with pytest.raises(ValueError):
            overlap_chunks(DOC, chunk_size=100, overlap=100)

    def test_strategy_metadata(self):
        chunks = overlap_chunks(DOC, chunk_size=100, overlap=10)
        assert all(c.metadata["strategy"] == "overlap" for c in chunks)


class TestParagraphChunks:
    def test_splits_on_blank_lines(self):
        chunks = paragraph_chunks(PARA_DOC, max_chars=1000)
        # All three paragraphs fit under 1000 chars → one chunk
        assert len(chunks) == 1

    def test_merges_short_paragraphs(self):
        chunks = paragraph_chunks(PARA_DOC, max_chars=1000)
        assert "First paragraph" in chunks[0].content
        assert "Second paragraph" in chunks[0].content

    def test_splits_when_over_max_chars(self):
        chunks = paragraph_chunks(PARA_DOC, max_chars=30)
        # Each paragraph is ~25 chars, so they should be split
        assert len(chunks) >= 2

    def test_strategy_metadata(self):
        chunks = paragraph_chunks(PARA_DOC, max_chars=1000)
        assert all(c.metadata["strategy"] == "paragraph" for c in chunks)


class TestChunkDocument:
    def test_raises_for_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_document(DOC, strategy="magic")

    def test_dispatches_overlap(self):
        chunks = chunk_document(DOC, strategy="overlap", chunk_size=100, overlap=10)
        assert all(c.metadata["strategy"] == "overlap" for c in chunks)


class TestChunkDocuments:
    def test_flattens_multiple_docs(self):
        docs = [DOC, PARA_DOC]
        chunks = chunk_documents(docs, strategy="fixed", chunk_size=50)
        assert len(chunks) > 2  # definitely more than 2 total chunks
