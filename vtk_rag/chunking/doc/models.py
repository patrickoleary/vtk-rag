"""Data class for API documentation chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocChunk:
    """Represents an API documentation chunk."""

    # Identifiers
    chunk_id: str
    chunk_type: str
    vtk_class: str

    # Semantic metadata
    action_phrase: str = ""
    synopsis: str = ""
    role: str = ""
    visibility_score: float = 0.5

    # Data types
    input_datatype: str = ""
    output_datatype: str = ""

    # Content
    content: str = ""

    # Module info
    module: str = ""

    # Queries for RAG retrieval
    queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "vtk_class": self.vtk_class,
            "content": self.content,
            "synopsis": self.synopsis,
            "role": self.role,
            "action_phrase": self.action_phrase,
            "visibility_score": self.visibility_score,
            "module": self.module,
            "input_datatype": self.input_datatype,
            "output_datatype": self.output_datatype,
            "queries": self.queries,
        }
