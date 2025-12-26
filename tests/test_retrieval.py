"""Tests for the retrieval module."""

from qdrant_client.models import Filter

from vtk_rag.config import load_config
from vtk_rag.rag import RAGClient


class TestRetriever:
    """Tests for the Retriever class."""

    def test_import(self):
        """Test that Retriever can be imported."""
        from vtk_rag.retrieval import Retriever
        assert Retriever is not None


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_import(self):
        """Test that SearchResult can be imported."""
        from vtk_rag.retrieval import SearchResult
        assert SearchResult is not None

    def test_create_search_result(self, sample_code_chunk: dict):
        """Test creating a SearchResult."""
        from vtk_rag.retrieval import SearchResult

        result = SearchResult(
            id=1,
            score=0.95,
            content=sample_code_chunk["content"],
            chunk_id=sample_code_chunk["chunk_id"],
            collection="vtk_code",
            payload=sample_code_chunk,
        )

        assert result.id == 1
        assert result.score == 0.95
        assert result.collection == "vtk_code"

    def test_search_result_properties(self, sample_code_chunk: dict):
        """Test SearchResult convenience properties."""
        from vtk_rag.retrieval import SearchResult

        result = SearchResult(
            id=1,
            score=0.95,
            content=sample_code_chunk["content"],
            chunk_id=sample_code_chunk["chunk_id"],
            collection="vtk_code",
            payload=sample_code_chunk,
        )

        assert result.class_name == "vtkSphereSource"
        assert result.synopsis == sample_code_chunk["synopsis"]
        assert result.role == "input"
        assert result.visibility_score == 0.9
        assert result.output_datatype == "vtkPolyData"

    def test_search_result_doc_properties(self, sample_doc_chunk: dict):
        """Test SearchResult properties for doc chunks."""
        from vtk_rag.retrieval import SearchResult

        result = SearchResult(
            id=2,
            score=0.85,
            content=sample_doc_chunk["content"],
            chunk_id=sample_doc_chunk["chunk_id"],
            collection="vtk_docs",
            payload=sample_doc_chunk,
        )

        assert result.class_name == "vtkSphereSource"
        assert result.chunk_type == "class_overview"
        assert result.action_phrase == "create a sphere"
        assert result.role == "input"
        assert result.module == "vtkmodules.vtkFiltersSources"


class TestRetrieverBuildFilter:
    """Tests for Retriever._build_filter method."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        from vtk_rag.retrieval import Retriever

        config = load_config()
        rag_client = RAGClient(config)
        retriever = Retriever(rag_client)

        result = retriever._build_filter(None)
        assert result is None

    def test_filter_passthrough(self):
        """Test that Filter objects pass through unchanged."""
        from vtk_rag.retrieval import Retriever

        config = load_config()
        rag_client = RAGClient(config)
        retriever = Retriever(rag_client)

        original = Filter(must=[])
        result = retriever._build_filter(original)
        assert result is original

    def test_exact_match(self):
        """Test exact match from dict."""
        from vtk_rag.retrieval import Retriever

        config = load_config()
        rag_client = RAGClient(config)
        retriever = Retriever(rag_client)

        result = retriever._build_filter({"role": "source_geometric"})
        assert result is not None
        assert len(result.must) == 1

    def test_match_any(self):
        """Test match-any from dict."""
        from vtk_rag.retrieval import Retriever

        config = load_config()
        rag_client = RAGClient(config)
        retriever = Retriever(rag_client)

        result = retriever._build_filter({
            "class_name": ["vtkSphereSource", "vtkConeSource"]
        })
        assert result is not None
        assert len(result.must) == 1

    def test_range(self):
        """Test range filter from dict."""
        from vtk_rag.retrieval import Retriever

        config = load_config()
        rag_client = RAGClient(config)
        retriever = Retriever(rag_client)

        result = retriever._build_filter({
            "visibility_score": {"gte": 0.7, "lte": 1.0}
        })
        assert result is not None
        assert len(result.must) == 1

    def test_combined(self):
        """Test combined filters from dict."""
        from vtk_rag.retrieval import Retriever

        config = load_config()
        rag_client = RAGClient(config)
        retriever = Retriever(rag_client)

        result = retriever._build_filter({
            "role": "source_geometric",
            "class_name": ["vtkSphereSource", "vtkConeSource"],
            "visibility_score": {"gte": 0.7},
        })
        assert result is not None
        assert len(result.must) == 3

    def test_empty_dict_returns_none(self):
        """Test that empty dict returns None."""
        from vtk_rag.retrieval import Retriever

        config = load_config()
        rag_client = RAGClient(config)
        retriever = Retriever(rag_client)

        result = retriever._build_filter({})
        assert result is None
