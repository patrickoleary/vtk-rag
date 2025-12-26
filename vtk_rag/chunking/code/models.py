"""CodeChunk dataclass for VTK code chunking.

Used by:
    SemanticChunk.build_chunk() in semantic_chunk.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeChunk:
    """Represents a semantic chunk from VTK code examples."""

    # Identifiers
    chunk_id: str
    example_id: str

    # Semantic metadata
    action_phrase: str
    synopsis: str
    role: str

    # Optional fields with defaults
    visibility_score: float = 0.5

    # Data types
    input_datatype: str = ""
    output_datatype: str = ""

    # Content
    content: str = ""

    # Programmatic information
    variable_name: str = ""
    vtk_classes: list[dict[str, Any]] = field(default_factory=list)

    # Queries for RAG retrieval
    queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization.

        Adds vtk_class_names (flat list) for Qdrant keyword indexing.
        """
        return {
            "chunk_id": self.chunk_id,
            "example_id": self.example_id,
            "action_phrase": self.action_phrase,
            "synopsis": self.synopsis,
            "role": self.role,
            "visibility_score": self.visibility_score,
            "input_datatype": self.input_datatype,
            "output_datatype": self.output_datatype,
            "content": self.content,
            "variable_name": self.variable_name,
            "vtk_classes": self.vtk_classes,
            "vtk_class_names": [c["class"] for c in self.vtk_classes if "class" in c],
            "queries": self.queries,
        }
