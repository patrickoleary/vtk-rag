"""Search result types for VTK RAG retrieval.

Used by:
    - retriever.py (Retriever.search)

Code Map:
    SearchResult
        from_qdrant()                # factory from Qdrant ScoredPoint
        class_name                   # property: VTK class name
        chunk_type                   # property: chunk type (docs only)
        synopsis                     # property: brief summary
        role                         # property: pipeline role
        example_id                   # property: source example URL
        variable_name                # property: primary variable
        input_datatype               # property: input data type
        output_datatype              # property: output data type
        visibility_score             # property: user-facing likelihood
        action_phrase                # property: action description
        module                       # property: VTK module path
        metadata                     # property: nested metadata dict
"""

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
        chunk_type: Chunk type (docs only).
        synopsis: Brief summary.
        role: Pipeline role (input, filters, properties, etc.).
        example_id: Source example URL (code only).
        variable_name: Primary variable (code only).
        input_datatype: Input data type.
        output_datatype: Output data type.
        visibility_score: User-facing likelihood.
        action_phrase: Concise action description.
        module: VTK module path (docs only).
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
        """VTK class name (first from vtk_class_names, or vtk_class for docs)."""
        # Code chunks: vtk_class_names is a list
        vtk_class_names = self.payload.get("vtk_class_names", [])
        if vtk_class_names:
            return vtk_class_names[0] if isinstance(vtk_class_names, list) else vtk_class_names
        # Doc chunks: vtk_class is a string
        return self.payload.get("vtk_class", "")

    @property
    def chunk_type(self) -> str:
        """Chunk type (docs only: class_overview, constructor, etc.)."""
        return self.payload.get("chunk_type", "")

    @property
    def synopsis(self) -> str:
        """Brief summary of the chunk."""
        return self.payload.get("synopsis", "")

    @property
    def role(self) -> str:
        """Pipeline role (input, filters, properties, renderer, scene, infrastructure, output, utility, color)."""
        return self.payload.get("role", "")

    @property
    def example_id(self) -> str:
        """Source example URL (code chunks only)."""
        return self.payload.get("example_id", "")

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
        """User-facing likelihood score (0.0-1.0)."""
        return self.payload.get("visibility_score", 0.0)

    @property
    def action_phrase(self) -> str:
        """Concise action description."""
        return self.payload.get("action_phrase", "")

    @property
    def module(self) -> str:
        """VTK module path (e.g., vtkmodules.vtkFiltersSources)."""
        return self.payload.get("module", "") or self.metadata.get("module", "")

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
