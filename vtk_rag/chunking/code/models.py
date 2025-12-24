"""Data models for VTK code chunking."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, TypedDict


class MethodCall(TypedDict):
    """Structure of a method call with its arguments."""
    name: str  # Method name (e.g., "SetRadius")
    args: list[str]  # String representations of arguments (e.g., ["10", "True"])


# VTKLifecycle represents a VTK object's usage within a code scope.
# Tracks the variable, class, statements, and relationships (mapper/actor/properties).
# Note: We use functional syntax to allow 'class' as a key (Python keyword).
VTKLifecycle = TypedDict('VTKLifecycle', {
    'variable': str | None,  # Variable name or None for static methods
    'class': str,  # VTK class name
    'type': str,  # Role classification (e.g., "properties", "infrastructure")
    'statements': list[ast.stmt],  # AST statements for this lifecycle
    'properties': list[dict[str, str]],  # Related properties with 'variable' and 'class' keys
    'mapper': str | None,  # Mapper variable if this is an actor
    'actor': str | None,  # Actor variable if this is a mapper
    'methods': list[str],  # Method names called on this VTK object (legacy, for compatibility)
    'method_calls': list[MethodCall],  # Method calls with arguments
}, total=False)


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
