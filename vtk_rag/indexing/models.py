"""Qdrant collection models."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FieldConfig:
    """Configuration for a payload field index."""

    # Field name in the chunk payload
    name: str

    # Qdrant index type (keyword, text, float, integer, bool)
    index_type: Literal["keyword", "text", "float", "integer", "bool"]

    # Human-readable description
    description: str = ""


@dataclass
class CollectionConfig:
    """Configuration for a Qdrant collection."""

    # Collection name in Qdrant
    name: str

    # Human-readable description
    description: str

    # Payload fields to index for filtering
    indexed_fields: list[FieldConfig] = field(default_factory=list)


CODE_CONFIG = CollectionConfig(
    name="vtk_code",
    description="VTK Python code chunks from examples and tests",
    indexed_fields=[
        FieldConfig("vtk_class_names", "keyword", "VTK class names"),
        FieldConfig("role", "keyword", "Pipeline role"),
        FieldConfig("input_datatype", "keyword", "Input data type"),
        FieldConfig("output_datatype", "keyword", "Output data type"),
        FieldConfig("example_id", "keyword", "Source example URL"),
        FieldConfig("variable_name", "keyword", "Variable name"),
        FieldConfig("visibility_score", "float", "User-facing likelihood (0.0-1.0)"),
        FieldConfig("action_phrase", "text", "Action description"),
        FieldConfig("synopsis", "text", "Summary with configuration details"),
    ]
)


DOC_CONFIG = CollectionConfig(
    name="vtk_docs",
    description="VTK API documentation chunks",
    indexed_fields=[
        FieldConfig("chunk_type", "keyword", "Chunk type"),
        FieldConfig("vtk_class", "keyword", "VTK class name"),
        FieldConfig("role", "keyword", "Pipeline role"),
        FieldConfig("module", "keyword", "VTK module path"),
        FieldConfig("input_datatype", "keyword", "Input data type"),
        FieldConfig("output_datatype", "keyword", "Output data type"),
        FieldConfig("visibility_score", "float", "User-facing likelihood (0.0-1.0)"),
        FieldConfig("action_phrase", "text", "Action description"),
        FieldConfig("synopsis", "text", "Brief summary"),
    ]
)
