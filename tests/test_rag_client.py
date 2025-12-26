"""Tests for the RAG client module."""

from vtk_rag.config import load_config
from vtk_rag.rag import RAGClient


class TestRAGClient:
    """Tests for the RAGClient class."""

    def test_import(self):
        """Test that RAGClient can be imported."""
        from vtk_rag.rag import RAGClient

        assert RAGClient is not None

    def test_init(self):
        """Test RAGClient initialization."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.config is config.rag_client
        assert rag_client.qdrant_client is not None
        assert rag_client.dense_model is not None
        assert rag_client.sparse_model is not None

    def test_qdrant_url_property(self):
        """Test qdrant_url property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.qdrant_url == config.rag_client.qdrant_url

    def test_code_collection_property(self):
        """Test code_collection property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.code_collection == config.rag_client.code_collection

    def test_docs_collection_property(self):
        """Test docs_collection property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.docs_collection == config.rag_client.docs_collection

    def test_top_k_property(self):
        """Test top_k property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.top_k == config.rag_client.top_k

    def test_use_hybrid_property(self):
        """Test use_hybrid property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.use_hybrid == config.rag_client.use_hybrid

    def test_min_visibility_score_property(self):
        """Test min_visibility_score property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.min_visibility_score == config.rag_client.min_visibility_score

    def test_raw_dir_property(self):
        """Test raw_dir property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.raw_dir == config.rag_client.raw_dir

    def test_chunk_dir_property(self):
        """Test chunk_dir property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.chunk_dir == config.rag_client.chunk_dir

    def test_examples_file_property(self):
        """Test examples_file property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.examples_file == config.rag_client.examples_file

    def test_tests_file_property(self):
        """Test tests_file property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.tests_file == config.rag_client.tests_file

    def test_docs_file_property(self):
        """Test docs_file property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.docs_file == config.rag_client.docs_file

    def test_code_chunks_property(self):
        """Test code_chunks property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.code_chunks == config.rag_client.code_chunks

    def test_doc_chunks_property(self):
        """Test doc_chunks property."""
        config = load_config()
        rag_client = RAGClient(config)

        assert rag_client.doc_chunks == config.rag_client.doc_chunks
