"""Tests for the chunking module."""

from pathlib import Path

import pytest


class TestChunker:
    """Tests for the Chunker class."""

    def test_import(self):
        """Test that Chunker can be imported."""
        from vtk_rag.chunking import Chunker
        assert Chunker is not None

    def test_init(self, mock_mcp_client):
        """Test Chunker initialization."""
        from vtk_rag.chunking import Chunker
        from vtk_rag.config import load_config
        from vtk_rag.rag import RAGClient

        config = load_config()
        rag_client = RAGClient(config)
        chunker = Chunker(rag_client, mock_mcp_client)
        assert chunker.base_path is not None

    def test_init_custom_path(self, tmp_path: Path, mock_mcp_client):
        """Test Chunker with custom base path."""
        from vtk_rag.chunking import Chunker
        from vtk_rag.config import load_config
        from vtk_rag.rag import RAGClient

        config = load_config()
        rag_client = RAGClient(config)
        chunker = Chunker(rag_client, mock_mcp_client, base_path=tmp_path)
        assert chunker.base_path == tmp_path


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client for testing."""
    class DummyMCPClient:
        def is_vtk_class(self, class_name):
            return class_name.startswith("vtk")
        def get_class_modules(self, class_names):
            return {}
        def get_class_info(self, class_name):
            return None
        def get_class_role(self, class_name):
            return "utility"
        def get_class_visibility(self, class_name):
            return "likely"
        def get_class_action_phrase(self, class_name):
            return None
        def get_method_signature(self, class_name, method_name):
            return None
        def get_class_synopsis(self, class_name):
            return None
        def get_class_doc(self, class_name):
            return None
    return DummyMCPClient()


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

    def test_semantic_query_import(self):
        """Test that SemanticQuery can be imported."""
        from vtk_rag.chunking.query import SemanticQuery
        assert SemanticQuery is not None

    def test_semantic_query_class_query(self):
        """Test class-level query generation."""
        from unittest.mock import MagicMock

        from vtk_rag.chunking.query import SemanticQuery
        mock_client = MagicMock()
        builder = SemanticQuery(mock_client)
        query = builder.class_to_query("isosurface generation")
        assert query == "How do you generate an isosurface?"

    def test_semantic_query_method_query(self):
        """Test method-level query generation."""
        from unittest.mock import MagicMock

        from vtk_rag.chunking.query import SemanticQuery
        mock_client = MagicMock()
        builder = SemanticQuery(mock_client)
        query = builder.method_to_query("SetRadius", "sphere")
        assert query == "How do you set the radius of a sphere?"

