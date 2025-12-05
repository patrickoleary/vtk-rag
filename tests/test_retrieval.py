"""Tests for the retrieval module."""



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
        assert result.chunk_type == "vtkmodules.vtkFiltersSources"
        assert result.synopsis == sample_code_chunk["synopsis"]
        assert result.title == sample_code_chunk["title"]
        assert result.role == "source_geometric"
        assert result.roles == ["source_geometric"]
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
        assert result.role == "source_geometric"


class TestFilterBuilder:
    """Tests for the FilterBuilder class."""

    def test_import(self):
        """Test that FilterBuilder can be imported."""
        from vtk_rag.retrieval import FilterBuilder
        assert FilterBuilder is not None

    def test_empty_build(self):
        """Test building with no conditions returns None."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder()
        result = builder.build()
        assert result is None

    def test_match(self):
        """Test exact match condition."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder().match("role", "source_geometric")
        result = builder.build()

        assert result is not None
        assert len(result.must) == 1

    def test_match_any(self):
        """Test match-any condition."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder().match_any("class_name", ["vtkSphereSource", "vtkConeSource"])
        result = builder.build()

        assert result is not None
        assert len(result.must) == 1

    def test_range(self):
        """Test range condition."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder().range("visibility_score", gte=0.7)
        result = builder.build()

        assert result is not None
        assert len(result.must) == 1

    def test_exclude(self):
        """Test exclusion condition."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder().exclude("chunk_type", "inheritance")
        result = builder.build()

        assert result is not None
        assert len(result.must_not) == 1

    def test_should_match(self):
        """Test should-match condition."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder().should_match("type", "Visualization Pipeline")
        result = builder.build()

        assert result is not None
        assert len(result.should) == 1

    def test_chaining(self):
        """Test method chaining."""
        from vtk_rag.retrieval import FilterBuilder

        builder = (
            FilterBuilder()
            .match("role", "source_geometric")
            .match_any("class_name", ["vtkSphereSource", "vtkConeSource"])
            .range("visibility_score", gte=0.7)
            .exclude("chunk_type", "inheritance")
        )
        result = builder.build()

        assert result is not None
        assert len(result.must) == 3
        assert len(result.must_not) == 1

    def test_from_dict_exact_match(self):
        """Test creating FilterBuilder from dict with exact match."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder.from_dict({"role": "source_geometric"})
        result = builder.build()

        assert result is not None
        assert len(result.must) == 1

    def test_from_dict_match_any(self):
        """Test creating FilterBuilder from dict with match-any."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder.from_dict({
            "class_name": ["vtkSphereSource", "vtkConeSource"]
        })
        result = builder.build()

        assert result is not None
        assert len(result.must) == 1

    def test_from_dict_range(self):
        """Test creating FilterBuilder from dict with range."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder.from_dict({
            "visibility_score": {"gte": 0.7, "lte": 1.0}
        })
        result = builder.build()

        assert result is not None
        assert len(result.must) == 1

    def test_from_dict_combined(self):
        """Test creating FilterBuilder from dict with multiple conditions."""
        from vtk_rag.retrieval import FilterBuilder

        builder = FilterBuilder.from_dict({
            "role": "source_geometric",
            "class_name": ["vtkSphereSource", "vtkConeSource"],
            "visibility_score": {"gte": 0.7},
        })
        result = builder.build()

        assert result is not None
        assert len(result.must) == 3
