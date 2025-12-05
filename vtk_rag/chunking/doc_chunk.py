"""Data class for API documentation chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocChunk:
    """Represents an API documentation chunk.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        chunk_type: Type of chunk ('class_overview', 'constructor',
                   'property_group', 'standalone_methods', 'inheritance').
        class_name: VTK class name (e.g., 'vtkSphereSource').
        content: Full text content of the chunk.
        synopsis: Brief summary of the chunk.
        role: Functional role (e.g., 'source_geometric', 'filter_general').
        action_phrase: Concise action description (e.g., 'create a sphere').
        visibility: User-facing likelihood ('very_likely', 'likely', 'maybe', etc.).
        metadata: Additional metadata (module, input/output datatypes, etc.).
        queries: Natural language queries for RAG retrieval.
    """
    chunk_id: str
    chunk_type: str
    class_name: str
    content: str
    synopsis: str
    role: str = ""
    action_phrase: str = ""
    visibility: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "class_name": self.class_name,
            "content": self.content,
            "synopsis": self.synopsis,
            "role": self.role,
            "action_phrase": self.action_phrase,
            "visibility": self.visibility,
            "metadata": self.metadata,
            "queries": self.queries,
        }
