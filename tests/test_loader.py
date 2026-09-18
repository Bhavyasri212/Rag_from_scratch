"""Unit tests for rag/loader.py"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.loader import _clean_text, load_document, load_documents, load_md, load_txt


class TestCleanText:
    def test_strips_whitespace(self):
        assert _clean_text("  hello  ") == "hello"

    def test_collapses_blank_lines(self):
        assert _clean_text("a\n\n\n\nb") == "a\n\nb"

    def test_normalises_windows_endings(self):
        assert "\r" not in _clean_text("a\r\nb")

    def test_collapses_inline_spaces(self):
        assert _clean_text("hello    world") == "hello world"

    def test_empty_string(self):
        assert _clean_text("") == ""


class TestLoadTxt:
    def test_returns_single_document(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("Hello world", encoding="utf-8")
        assert len(load_txt(f)) == 1

    def test_content_is_cleaned(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("  Hello   World  \r\n\r\n", encoding="utf-8")
        assert load_txt(f)[0]["content"] == "Hello World"

    def test_metadata_file_type(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("x", encoding="utf-8")
        assert load_txt(f)[0]["metadata"]["file_type"] == "txt"

    def test_char_count_matches_content(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("Hello!", encoding="utf-8")
        doc = load_txt(f)[0]
        assert doc["metadata"]["char_count"] == len(doc["content"])


class TestLoadMd:
    def test_preserves_markdown_syntax(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text("# Heading\n**bold**", encoding="utf-8")
        doc = load_md(f)[0]
        assert "# Heading" in doc["content"]
        assert "**bold**" in doc["content"]

    def test_file_type_is_md(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text("# T", encoding="utf-8")
        assert load_md(f)[0]["metadata"]["file_type"] == "md"


class TestLoadDocument:
    def test_dispatches_txt(self, tmp_path):
        f = tmp_path / "s.txt"
        f.write_text("hi", encoding="utf-8")
        assert load_document(f)[0]["metadata"]["file_type"] == "txt"

    def test_dispatches_md(self, tmp_path):
        f = tmp_path / "s.md"
        f.write_text("# hi", encoding="utf-8")
        assert load_document(f)[0]["metadata"]["file_type"] == "md"

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "nope.txt")

    def test_raises_for_unsupported_extension(self, tmp_path):
        f = tmp_path / "file.docx"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(f)


class TestLoadDocuments:
    def test_flattens_multiple_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        (tmp_path / "b.md").write_text("B", encoding="utf-8")
        assert len(load_documents([tmp_path / "a.txt", tmp_path / "b.md"])) == 2

    def test_empty_list(self):
        assert load_documents([]) == []
