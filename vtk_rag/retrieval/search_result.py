"""Search result types for VTK RAG retrieval."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """A single search result from Qdrant.

    Attributes:
        id: Qdrant point ID.
        score: Relevance score (higher is better).
        content: Chunk content text.
        chunk_id: Original chunk identifier.
        collection: Source collection (vtk_code or vtk_docs).
        payload: Full payload from Qdrant (all metadata fields).

    Properties:
        class_name: VTK class name.
        chunk_type: Chunk type.
        synopsis: Brief summary.
        title: Human-readable title (code).
        role: Primary functional role.
        roles: All functional roles (code).
        description: Detailed description (code).
        example_id: Source example URL (code).
        function_name: Containing function (code).
        variable_name: Primary variable (code).
        input_datatype: Input data type.
        output_datatype: Output data type.
        visibility_score: User-facing likelihood (code).
        action_phrase: Concise action description (docs).
        module: VTK module path.
        metadata: Nested metadata dict.
    """
    id: int
    score: float
    content: str
    chunk_id: str
    collection: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        """VTK class name (from vtk_class or class_name field)."""
        return self.payload.get("vtk_class") or self.payload.get("class_name", "")

    @property
    def chunk_type(self) -> str:
        """Chunk type (type for code, chunk_type for docs)."""
        return self.payload.get("type") or self.payload.get("chunk_type", "")

    @property
    def synopsis(self) -> str:
        """Brief summary of the chunk."""
        return self.payload.get("synopsis", "")

    @property
    def title(self) -> str:
        """Human-readable title (code chunks only)."""
        return self.payload.get("title", "")

    @property
    def role(self) -> str:
        """Functional role (e.g., source_geometric, filter_general)."""
        return self.payload.get("role") or self.payload.get("roles", [""])[0] if self.payload.get("roles") else ""

    @property
    def roles(self) -> list[str]:
        """All functional roles (code chunks may have multiple)."""
        return self.payload.get("roles", [])

    @property
    def description(self) -> str:
        """Detailed description (code chunks only)."""
        return self.payload.get("description", "")

    @property
    def example_id(self) -> str:
        """Source example URL (code chunks only)."""
        return self.payload.get("example_id", "")

    @property
    def function_name(self) -> str:
        """Containing function name (code chunks only)."""
        return self.payload.get("function_name", "")

    @property
    def variable_name(self) -> str:
        """Primary variable name (code chunks only)."""
        return self.payload.get("variable_name", "")

    @property
    def input_datatype(self) -> str:
        """Input data type (e.g., vtkPolyData)."""
        return self.payload.get("input_datatype", "") or self.metadata.get("input_datatype", "")

    @property
    def output_datatype(self) -> str:
        """Output data type (e.g., vtkPolyData)."""
        return self.payload.get("output_datatype", "") or self.metadata.get("output_datatype", "")

    @property
    def visibility_score(self) -> float:
        """User-facing likelihood score (0.0-1.0, code chunks only)."""
        return self.payload.get("visibility_score", 0.0)

    @property
    def action_phrase(self) -> str:
        """Concise action description (doc chunks only)."""
        return self.payload.get("action_phrase", "")

    @property
    def module(self) -> str:
        """VTK module path (e.g., vtkmodules.vtkFiltersSources)."""
        return self.metadata.get("module", "")

    @property
    def metadata(self) -> dict[str, Any]:
        """Nested metadata dict."""
        return self.payload.get("metadata", {})

    @classmethod
    def from_qdrant(cls, point: Any, collection: str) -> "SearchResult":
        """Create SearchResult from Qdrant ScoredPoint.

        Args:
            point: Qdrant ScoredPoint object.
            collection: Collection name.

        Returns:
            SearchResult instance.
        """
        payload = point.payload or {}
        return cls(
            id=point.id,
            score=point.score,
            content=payload.get("content", ""),
            chunk_id=payload.get("chunk_id", ""),
            collection=collection,
            payload=payload,
        )
