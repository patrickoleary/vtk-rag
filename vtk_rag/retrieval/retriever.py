"""Core retrieval class for VTK RAG.

Provides search operations over Qdrant collections with support for
semantic, BM25, and hybrid search modes.
"""

from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)
from sentence_transformers import SentenceTransformer

from .filter_builder import FilterBuilder
from .search_result import SearchResult


class Retriever:
    """Search VTK code and documentation in Qdrant.

    Supports:
    - Semantic search (dense vectors)
    - BM25 search (sparse vectors)
    - Hybrid search (dense + sparse with RRF fusion)
    - Multi-vector search (pre-generated queries)
    - Metadata filtering

    Example:
        retriever = Retriever()

        # Semantic search (dense vectors)
        results = retriever.search("create a sphere", collection="vtk_code")

        # BM25 search (sparse vectors)
        results = retriever.bm25_search("vtkSphereSource", collection="vtk_code")

        # Hybrid search (dense + sparse with RRF fusion)
        results = retriever.hybrid_search("vtkSphereSource", collection="vtk_docs")

        # Multi-vector search (pre-generated queries)
        results = retriever.search("sphere", vector_name="queries")

        # Filtered search
        results = retriever.search(
            "render pipeline",
            collection="vtk_code",
            filters={"role": "source_geometric"},
        )

        # Convenience methods
        results = retriever.search_code("render pipeline")
        results = retriever.search_docs("vtkPolyDataMapper")
        results = retriever.search_by_class("vtkSphereSource")
        results = retriever.search_by_role("create geometry", role="source_geometric")
        results = retriever.search_by_datatype("filter mesh", input_type="vtkPolyData")
        results = retriever.search_by_module("sources", module="vtkFiltersSources")
        results = retriever.search_by_chunk_type("API", chunk_type="class_overview")
    """

    # Collection names
    CODE_COLLECTION = "vtk_code"
    DOCS_COLLECTION = "vtk_docs"

    def __init__(
        self,
        qdrant_url: str | None = None,
        dense_model: str | None = None,
        sparse_model: str | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            qdrant_url: Qdrant server URL. Defaults to config or localhost:6333.
            dense_model: SentenceTransformer model for dense embeddings.
            sparse_model: FastEmbed model for sparse (BM25) embeddings.
        """
        # Load config for defaults
        from vtk_rag.config import get_config
        config = get_config()
        
        self.client = QdrantClient(url=qdrant_url or config.qdrant.url)
        self.dense_model = SentenceTransformer(dense_model or config.embedding.dense_model)
        self.sparse_model = SparseTextEmbedding(sparse_model or config.embedding.sparse_model)

    def search(
        self,
        query: str,
        collection: str = "vtk_code",
        limit: int = 10,
        filters: dict[str, Any] | Filter | None = None,
        vector_name: str = "content",
    ) -> list[SearchResult]:
        """Semantic search using dense vectors.

        Args:
            query: Natural language query.
            collection: Collection to search (vtk_code or vtk_docs).
            limit: Maximum results to return.
            filters: Filter conditions as dict or Qdrant Filter. Dict format:
                - Exact match: {"field": "value"}
                - Match any: {"field": ["val1", "val2"]}
                - Range: {"field": {"gte": 0.7, "lt": 1.0}}

                Code collection fields: type, vtk_class, function_name, roles,
                    input_datatype, output_datatype, example_id, variable_name,
                    visibility_score, title, synopsis.

                Doc collection fields: chunk_type, class_name, role, visibility,
                    metadata.module, metadata.input_datatype, metadata.output_datatype,
                    action_phrase, synopsis.
            vector_name: Vector to search ("content" for semantic, "queries" for
                pre-generated query embeddings).

        Returns:
            List of SearchResult objects sorted by relevance score.
        """
        # Generate query embedding
        query_vector = self.dense_model.encode(query).tolist()

        # Build filter
        query_filter = self._build_filter(filters)

        # Execute search
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
        )

        return [SearchResult.from_qdrant(r, collection) for r in results.points]

    def bm25_search(
        self,
        query: str,
        collection: str = "vtk_code",
        limit: int = 10,
        filters: dict[str, Any] | Filter | None = None,
    ) -> list[SearchResult]:
        """BM25 keyword search using sparse vectors.

        Args:
            query: Search query (keywords work best).
            collection: Collection to search.
            limit: Maximum results to return.
            filters: Filter conditions as dict or Qdrant Filter. Dict format:
                - Exact match: {"field": "value"}
                - Match any: {"field": ["val1", "val2"]}
                - Range: {"field": {"gte": 0.7, "lt": 1.0}}
                See search() docstring for available fields per collection.

        Returns:
            List of SearchResult objects.
        """
        # Generate sparse embedding
        sparse_emb = list(self.sparse_model.embed([query]))[0]
        sparse_vector = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        # Build filter
        query_filter = self._build_filter(filters)

        # Execute search
        results = self.client.query_points(
            collection_name=collection,
            query=sparse_vector,
            using="bm25",
            query_filter=query_filter,
            limit=limit,
        )

        return [SearchResult.from_qdrant(r, collection) for r in results.points]

    def hybrid_search(
        self,
        query: str,
        collection: str = "vtk_code",
        limit: int = 10,
        filters: dict[str, Any] | Filter | None = None,
        prefetch_limit: int = 20,
    ) -> list[SearchResult]:
        """Hybrid search combining dense and sparse vectors with RRF fusion.

        Best for queries that mix natural language with VTK class names.

        Args:
            query: Search query.
            collection: Collection to search.
            limit: Maximum results to return.
            filters: Filter conditions as dict or Qdrant Filter. Dict format:
                - Exact match: {"field": "value"}
                - Match any: {"field": ["val1", "val2"]}
                - Range: {"field": {"gte": 0.7, "lt": 1.0}}
                See search() docstring for available fields per collection.
            prefetch_limit: Number of results to prefetch from each vector.

        Returns:
            List of SearchResult objects.
        """
        # Generate embeddings
        dense_vector = self.dense_model.encode(query).tolist()
        sparse_emb = list(self.sparse_model.embed([query]))[0]
        sparse_vector = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        # Build filter
        query_filter = self._build_filter(filters)

        # Execute hybrid search with RRF fusion
        results = self.client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="content",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="bm25",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )

        return [SearchResult.from_qdrant(r, collection) for r in results.points]

    def search_code(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        hybrid: bool = True,
    ) -> list[SearchResult]:
        """Search code chunks.

        Convenience method for searching vtk_code collection.

        Args:
            query: Search query.
            limit: Maximum results.
            filters: Filter conditions as dict. Dict format:
                - Exact match: {"field": "value"}
                - Match any: {"field": ["val1", "val2"]}
                - Range: {"field": {"gte": 0.7, "lt": 1.0}}

                Available fields: type, vtk_class, function_name, roles,
                    input_datatype, output_datatype, example_id, variable_name,
                    visibility_score, title, synopsis.
            hybrid: Use hybrid search (default True).

        Returns:
            List of SearchResult objects.
        """
        if hybrid:
            return self.hybrid_search(query, self.CODE_COLLECTION, limit, filters)
        return self.search(query, self.CODE_COLLECTION, limit, filters)

    def search_docs(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        hybrid: bool = True,
    ) -> list[SearchResult]:
        """Search documentation chunks.

        Convenience method for searching vtk_docs collection.

        Args:
            query: Search query.
            limit: Maximum results.
            filters: Filter conditions as dict. Dict format:
                - Exact match: {"field": "value"}
                - Match any: {"field": ["val1", "val2"]}
                - Range: {"field": {"gte": 0.7, "lt": 1.0}}

                Available fields: chunk_type, class_name, role, visibility,
                    metadata.module, metadata.input_datatype, metadata.output_datatype,
                    action_phrase, synopsis.
            hybrid: Use hybrid search (default True).

        Returns:
            List of SearchResult objects.
        """
        if hybrid:
            return self.hybrid_search(query, self.DOCS_COLLECTION, limit, filters)
        return self.search(query, self.DOCS_COLLECTION, limit, filters)

    def search_by_class(
        self,
        class_name: str,
        collection: str = "vtk_docs",
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for chunks related to a specific VTK class.

        Uses BM25 for exact class name matching.

        Args:
            class_name: VTK class name (e.g., "vtkSphereSource").
            collection: Collection to search.
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        field = "class_name" if collection == self.DOCS_COLLECTION else "vtk_class"
        return self.bm25_search(
            query=class_name,
            collection=collection,
            limit=limit,
            filters={field: class_name},
        )

    def search_by_role(
        self,
        query: str,
        role: str,
        collection: str = "vtk_code",
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for chunks with a specific functional role.

        Args:
            query: Search query.
            role: Functional role (e.g., "source_geometric", "filter_general",
                "mapper_polydata", "actor", "renderer").
            collection: Collection to search.
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        field = "role" if collection == self.DOCS_COLLECTION else "roles"
        return self.hybrid_search(
            query=query,
            collection=collection,
            limit=limit,
            filters={field: role},
        )

    def search_by_datatype(
        self,
        query: str,
        input_type: str | None = None,
        output_type: str | None = None,
        collection: str = "vtk_code",
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for chunks by input/output data type.

        Args:
            query: Search query.
            input_type: Input data type (e.g., "vtkPolyData", "vtkImageData").
            output_type: Output data type.
            collection: Collection to search.
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        filters: dict[str, Any] = {}
        if collection == self.DOCS_COLLECTION:
            if input_type:
                filters["metadata.input_datatype"] = input_type
            if output_type:
                filters["metadata.output_datatype"] = output_type
        else:
            if input_type:
                filters["input_datatype"] = input_type
            if output_type:
                filters["output_datatype"] = output_type

        return self.hybrid_search(
            query=query,
            collection=collection,
            limit=limit,
            filters=filters if filters else None,
        )

    def search_by_module(
        self,
        query: str,
        module: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search documentation by VTK module.

        Args:
            query: Search query.
            module: VTK module path (e.g., "vtkFiltersSources", "vtkRenderingCore").
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        return self.hybrid_search(
            query=query,
            collection=self.DOCS_COLLECTION,
            limit=limit,
            filters={"metadata.module": module},
        )

    def search_by_chunk_type(
        self,
        query: str,
        chunk_type: str,
        collection: str = "vtk_docs",
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for specific chunk types.

        Args:
            query: Search query.
            chunk_type: For docs: "class_overview", "constructor", "property_group",
                "standalone_methods", "inheritance".
                For code: "Visualization Pipeline", "Rendering Infrastructure", etc.
            collection: Collection to search.
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        field = "chunk_type" if collection == self.DOCS_COLLECTION else "type"
        return self.hybrid_search(
            query=query,
            collection=collection,
            limit=limit,
            filters={field: chunk_type},
        )

    def _build_filter(
        self,
        filters: dict[str, Any] | Filter | None,
    ) -> Filter | None:
        """Convert filter input to Qdrant Filter.

        Args:
            filters: Dict, Filter, or None.

        Returns:
            Qdrant Filter or None.
        """
        if filters is None:
            return None
        if isinstance(filters, Filter):
            return filters
        return FilterBuilder.from_dict(filters).build()
