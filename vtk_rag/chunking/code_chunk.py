"""Data class for VTK code example chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeChunk:
    """Represents a semantic chunk from VTK code examples.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        example_id: ID of the source example file.
        type: Chunk type ('Visualization Pipeline', 'Rendering Infrastructure',
              'Actors', 'Mappers', etc.).
        function_name: Name of the containing function/method.
        title: Human-readable title for the chunk.
        description: Detailed description of the chunk.
        synopsis: Brief summary of the chunk.
        roles: List of VTK class roles in this chunk.
        visibility_score: Computed visibility score (0.0-1.0).
        input_datatype: Input data type (e.g., 'vtkPolyData').
        output_datatype: Output data type (e.g., 'vtkPolyData').
        content: Full code content of the chunk.
        metadata: Additional metadata (class_context, etc.).
        variable_name: Variable name for single-lifecycle chunks.
        vtk_class: VTK class name for single-lifecycle chunks.
        queries: Natural language queries for RAG retrieval.
    """
    chunk_id: str
    example_id: str
    type: str
    function_name: str
    title: str
    description: str
    synopsis: str
    roles: list[str] = field(default_factory=list)
    visibility_score: float = 0.5
    input_datatype: str = "N/A"
    output_datatype: str = "N/A"
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    variable_name: str = "N/A"
    vtk_class: str = "N/A"
    queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "example_id": self.example_id,
            "type": self.type,
            "function_name": self.function_name,
            "title": self.title,
            "description": self.description,
            "synopsis": self.synopsis,
            "roles": self.roles,
            "visibility_score": self.visibility_score,
            "input_datatype": self.input_datatype,
            "output_datatype": self.output_datatype,
            "content": self.content,
            "metadata": self.metadata,
            "variable_name": self.variable_name,
            "vtk_class": self.vtk_class,
            "queries": self.queries,
        }
