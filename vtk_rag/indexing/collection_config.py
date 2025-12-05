"""Collection configuration for Qdrant indexes.

Qdrant Search Architecture
==========================

Each collection supports three search modes that can be combined:

1. DENSE VECTORS (Semantic Search)
   - Model: SentenceTransformer (all-MiniLM-L6-v2)
   - Finds chunks by meaning similarity
   - "content" vector: single 384-dim embedding of chunk text
   - "queries" vector: multi-vector of pre-generated query embeddings

2. SPARSE VECTORS (BM25 Keyword Search)
   - Model: FastEmbed (Qdrant/bm25)
   - Finds chunks by exact term matching with IDF weighting
   - "bm25" vector: sparse embedding of chunk text
   - Good for VTK class names like "vtkSphereSource"

3. PAYLOAD INDEXES (Metadata Filtering)
   - keyword: exact match (e.g., class_name="vtkConeSource")
   - text: tokenized full-text search
   - float/integer: range queries (e.g., visibility_score >= 0.7)

Hybrid Search
-------------
Combine dense + sparse with Reciprocal Rank Fusion (RRF):
- Prefetch top-N from each vector type
- Fuse rankings to get best of both worlds
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FieldConfig:
    """Configuration for a payload field index.

    Payload indexes enable filtering during vector search. Qdrant creates
    these indexes on document metadata fields.

    Attributes:
        name: Field name in the chunk payload (e.g., "class_name").
        index_type: Qdrant index type for this field.
        description: Human-readable description.

    Index Types:
        keyword: Exact match on string values. Fast for filtering.
        text: Tokenized full-text search. Slower but supports partial matches.
        float/integer: Numeric range queries.
        bool: Boolean filtering.
        geo: Geographic point queries.
    """
    name: str
    index_type: Literal["keyword", "text", "float", "integer", "bool", "geo"]
    description: str = ""


@dataclass
class CollectionConfig:
    """Configuration for a Qdrant collection.

    Defines the schema for vectors and payload indexes. The Indexer uses
    this to create collections with the correct structure.

    Attributes:
        name: Collection name in Qdrant (e.g., "vtk_code").
        description: Human-readable description.
        vector_field: Name of the dense content vector.
        query_vector_field: Name of the multi-vector for pre-generated queries.
        indexed_fields: List of payload fields to index for filtering.
    """
    name: str
    description: str
    vector_field: str = "content"
    query_vector_field: str = "queries"
    indexed_fields: list[FieldConfig] = field(default_factory=list)

    def get_field_schemas(self) -> dict[str, str]:
        """Return field name to schema type mapping for Qdrant."""
        return {f.name: f.index_type for f in self.indexed_fields}


# =============================================================================
# Code Collection Configuration
# =============================================================================
#
# Vectors created by Indexer:
#   - "content": Dense 384-dim vector from SentenceTransformer
#   - "queries": Multi-vector of pre-generated query embeddings
#   - "bm25": Sparse vector from FastEmbed BM25 model
#
# Payload indexes below enable filtering during search.

CODE_COLLECTION_CONFIG = CollectionConfig(
    name="vtk_code",
    description="VTK Python code chunks from examples and tests",
    vector_field="content",
    query_vector_field="queries",
    indexed_fields=[
        # Keyword filters (exact match) - fast for filtering
        FieldConfig("type", "keyword", "Chunk type: Visualization Pipeline, Rendering Infrastructure, vtkmodules.*"),
        FieldConfig("vtk_class", "keyword", "Primary VTK class (e.g., vtkCylinderSource)"),
        FieldConfig("function_name", "keyword", "Containing function name"),
        FieldConfig("roles", "keyword", "Functional roles (source_geometric, mapper_polydata, etc.)"),
        FieldConfig("input_datatype", "keyword", "Input data type (vtkPolyData, vtkImageData, etc.)"),
        FieldConfig("output_datatype", "keyword", "Output data type"),
        FieldConfig("example_id", "keyword", "Source example URL"),
        FieldConfig("variable_name", "keyword", "Variable name for single-lifecycle chunks"),

        # Numeric filters - for range queries
        FieldConfig("visibility_score", "float", "User-facing likelihood (0.0-1.0)"),

        # Text indexes - tokenized for partial matching (separate from BM25 sparse vector)
        FieldConfig("title", "text", "Human-readable title"),
        FieldConfig("synopsis", "text", "Natural language summary"),
    ]
)


# =============================================================================
# Doc Collection Configuration
# =============================================================================
#
# Vectors created by Indexer:
#   - "content": Dense 384-dim vector from SentenceTransformer
#   - "queries": Multi-vector of pre-generated query embeddings
#   - "bm25": Sparse vector from FastEmbed BM25 model
#
# Payload indexes below enable filtering during search.

DOC_COLLECTION_CONFIG = CollectionConfig(
    name="vtk_docs",
    description="VTK API documentation chunks",
    vector_field="content",
    query_vector_field="queries",
    indexed_fields=[
        # Keyword filters (exact match) - fast for filtering
        FieldConfig("chunk_type", "keyword", "Chunk type: class_overview, constructor, property_group, standalone_methods, inheritance"),
        FieldConfig("class_name", "keyword", "VTK class name (e.g., vtkSphereSource)"),
        FieldConfig("role", "keyword", "Functional role (source_geometric, filter_general, etc.)"),
        FieldConfig("visibility", "keyword", "User-facing likelihood (very_likely, likely, maybe, etc.)"),
        FieldConfig("metadata.module", "keyword", "VTK module path"),
        FieldConfig("metadata.input_datatype", "keyword", "Input data type"),
        FieldConfig("metadata.output_datatype", "keyword", "Output data type"),

        # Text indexes - tokenized for partial matching (separate from BM25 sparse vector)
        FieldConfig("action_phrase", "text", "Concise action description"),
        FieldConfig("synopsis", "text", "Brief summary"),
    ]
)
