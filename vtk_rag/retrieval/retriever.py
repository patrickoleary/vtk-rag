"""Core retrieval class for VTK RAG.

Provides search operations over Qdrant collections with support for
semantic, BM25, and hybrid search modes.

Used by:
    - cli.py (cmd_search)

Code Map:
    Retriever
        search()                     # semantic search (dense vectors)
        bm25_search()                # keyword search (sparse vectors)
        hybrid_search()              # combined dense + sparse with RRF
        search_code()                # convenience: search code collection
        search_docs()                # convenience: search docs collection
        search_by_class()            # filter by VTK class name
        search_by_role()             # filter by pipeline role
        search_by_datatype()         # filter by input/output type
        search_by_module()           # filter by VTK module
        search_by_chunk_type()       # filter by chunk type (docs)
        _build_filter()              # convert dict to Qdrant Filter
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    Range,
    SparseVector,
)

from vtk_rag.rag import RAGClient

from .models import SearchResult


class Retriever:
    """Search VTK code and documentation in Qdrant.

    Attributes:
        rag_client: RAG client providing Qdrant and embedding access.
        qdrant_client: Qdrant client for vector database operations.
    """

    CODE_COLLECTION: str = "vtk_code"
    DOCS_COLLECTION: str = "vtk_docs"

    rag_client: RAGClient
    qdrant_client: QdrantClient

    def __init__(self, rag_client: RAGClient) -> None:
        """Initialize the retriever.

        Args:
            rag_client: RAG client providing Qdrant and embedding access.
        """
        self.rag_client = rag_client
        self.qdrant_client = rag_client.qdrant_client

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
            filters: Filter conditions as dict or Qdrant Filter.
            vector_name: Vector to search (content or queries).

        Returns:
            List of SearchResult objects sorted by relevance score.
        """
        # Generate query embedding
        query_vector = self.rag_client.dense_model.encode(query).tolist()

        # Build filter
        query_filter = self._build_filter(filters)

        # Execute search
        results = self.qdrant_client.query_points(
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
            filters: Filter conditions as dict or Qdrant Filter.

        Returns:
            List of SearchResult objects.
        """
        # Generate sparse embedding
        sparse_emb = list(self.rag_client.sparse_model.embed([query]))[0]
        sparse_vector = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        # Build filter
        query_filter = self._build_filter(filters)

        # Execute search
        results = self.qdrant_client.query_points(
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

        Args:
            query: Search query.
            collection: Collection to search.
            limit: Maximum results to return.
            filters: Filter conditions as dict or Qdrant Filter.
            prefetch_limit: Number of results to prefetch from each vector.

        Returns:
            List of SearchResult objects.
        """
        # Generate embeddings
        dense_vector = self.rag_client.dense_model.encode(query).tolist()
        sparse_emb = list(self.rag_client.sparse_model.embed([query]))[0]
        sparse_vector = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        # Build filter
        query_filter = self._build_filter(filters)

        # Execute hybrid search with RRF fusion
        results = self.qdrant_client.query_points(
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

        Args:
            query: Search query.
            limit: Maximum results.
            filters: Filter conditions as dict.
            hybrid: Use hybrid search.

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

        Args:
            query: Search query.
            limit: Maximum results.
            filters: Filter conditions as dict.
            hybrid: Use hybrid search.

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

        Args:
            class_name: VTK class name.
            collection: Collection to search.
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        return self.bm25_search(
            query=class_name,
            collection=collection,
            limit=limit,
            filters={"vtk_class_names": class_name},
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
            role: Functional role.
            collection: Collection to search.
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        field = "role"
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
            input_type: Input data type.
            output_type: Output data type.
            collection: Collection to search.
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        filters: dict[str, Any] = {}
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
            module: VTK module path.
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        return self.hybrid_search(
            query=query,
            collection=self.DOCS_COLLECTION,
            limit=limit,
            filters={"module": module},
        )

    def search_by_chunk_type(
        self,
        query: str,
        chunk_type: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search documentation by chunk type.

        Args:
            query: Search query.
            chunk_type: Chunk type.
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        return self.hybrid_search(
            query=query,
            collection=self.DOCS_COLLECTION,
            limit=limit,
            filters={"chunk_type": chunk_type},
        )

    def _build_filter(
        self,
        filters: dict[str, Any] | Filter | None,
    ) -> Filter | None:
        """Convert filter dict to Qdrant Filter.

        Args:
            filters: Dict with filter conditions, Qdrant Filter, or None.

                Code collection fields:
                    vtk_class_names, role, input_datatype,
                    output_datatype, example_id, variable_name,
                    visibility_score (float), action_phrase, synopsis.

                Doc collection fields:
                    chunk_type, vtk_class, role, module, input_datatype,
                    output_datatype, visibility_score (float),
                    action_phrase, synopsis.

                Dict formats:
                    Exact match: {"vtk_class_names": "vtkSphereSource"}
                    Match any: {"role": ["source", "filter"]}
                    Range: {"visibility_score": {"gte": 0.7}}
                    Example: {"vtk_class_names": "vtkActor",
                               "role": "actor",
                               "visibility_score": {"gte": 0.7}}

        Returns:
            Qdrant Filter or None.
        """
        if filters is None:
            return None
        if isinstance(filters, Filter):
            return filters

        conditions = []
        for field, value in filters.items():
            if isinstance(value, dict):
                # Range filter
                conditions.append(
                    FieldCondition(
                        key=field,
                        range=Range(
                            gt=value.get("gt"),
                            gte=value.get("gte"),
                            lt=value.get("lt"),
                            lte=value.get("lte"),
                        ),
                    )
                )
            elif isinstance(value, list):
                # Match any
                conditions.append(
                    FieldCondition(key=field, match=MatchAny(any=value))
                )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None
