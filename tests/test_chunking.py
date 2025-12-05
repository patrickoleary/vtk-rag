"""Tests for the chunking module."""

from pathlib import Path


class TestChunker:
    """Tests for the Chunker class."""

    def test_import(self):
        """Test that Chunker can be imported."""
        from vtk_rag.chunking import Chunker
        assert Chunker is not None

    def test_init(self):
        """Test Chunker initialization."""
        from vtk_rag.chunking import Chunker

        chunker = Chunker()
        assert chunker.base_path is not None

    def test_init_custom_path(self, tmp_path: Path):
        """Test Chunker with custom base path."""
        from vtk_rag.chunking import Chunker

        chunker = Chunker(base_path=tmp_path)
        assert chunker.base_path == tmp_path


class TestCodeChunker:
    """Tests for the CodeChunker class."""

    def test_import(self):
        """Test that CodeChunker can be imported."""
        from vtk_rag.chunking import CodeChunker
        assert CodeChunker is not None


class TestDocChunker:
    """Tests for the DocChunker class."""

    def test_import(self):
        """Test that DocChunker can be imported."""
        from vtk_rag.chunking import DocChunker
        assert DocChunker is not None


class TestQueryGenerators:
    """Tests for query generator classes."""

    def test_code_query_generator_import(self):
        """Test that CodeQueryGenerator can be imported."""
        from vtk_rag.chunking import CodeQueryGenerator
        assert CodeQueryGenerator is not None

    def test_doc_query_generator_import(self):
        """Test that DocQueryGenerator can be imported."""
        from vtk_rag.chunking import DocQueryGenerator
        assert DocQueryGenerator is not None
