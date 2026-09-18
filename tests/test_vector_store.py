"""Unit tests for rag/vector_store.py"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.chunker import Chunk
from rag.vector_store import VectorStore, ScoredChunk, cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_return_1(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_0(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_1(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_0(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_similarity(a, b) == 0.0


class TestVectorStore:
    def _make_chunk(self, text="hello"):
        return Chunk(content=text, metadata={"source": "test.txt"})

    def _make_store(self, n=3):
        store = VectorStore()
        for i in range(n):
            v = np.zeros(4)
            v[i % 4] = 1.0  # unit vectors in different directions
            store.add(self._make_chunk(f"chunk {i}"), v)
        return store

    def test_len_grows_on_add(self):
        store = VectorStore()
        assert len(store) == 0
        store.add(self._make_chunk(), np.array([1.0, 0.0]))
        assert len(store) == 1

    def test_search_returns_k_results(self):
        store = self._make_store(n=5)
        results = store.search(np.array([1.0, 0.0, 0.0, 0.0]), k=3)
        assert len(results) == 3

    def test_top_result_is_most_similar(self):
        store = self._make_store(n=3)
        query = np.array([1.0, 0.0, 0.0, 0.0])  # points in direction of chunk 0
        results = store.search(query, k=3)
        assert results[0].chunk.content == "chunk 0"
        assert results[0].score == pytest.approx(1.0)

    def test_search_empty_store_returns_empty(self):
        store = VectorStore()
        assert store.search(np.array([1.0, 0.0]), k=3) == []

    def test_add_batch(self):
        store = VectorStore()
        chunks = [self._make_chunk(f"c{i}") for i in range(4)]
        vecs = [np.eye(4)[i] for i in range(4)]
        store.add_batch(chunks, vecs)
        assert len(store) == 4

    def test_save_and_load(self, tmp_path):
        store = self._make_store(n=2)
        p = tmp_path / "store.json"
        store.save(p)

        store2 = VectorStore()
        store2.load(p)
        assert len(store2) == 2
        assert store2._chunks[0].content == store._chunks[0].content
