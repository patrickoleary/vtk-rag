# VTK RAG

Retrieval-Augmented Generation for VTK code and documentation.

Transform natural language queries into relevant VTK code examples and class/method documentation using semantic search with hybrid vector + BM25 indexing.

## Quick Start

```bash
# Setup (installs uv if needed, creates .venv, installs dependencies)
./setup.sh --dev

# Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Build (chunk + index)
uv run vtk-rag build

# Search
uv run vtk-rag search "create a sphere"
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

### Option 1: Using setup.sh (Recommended)

```bash
./setup.sh          # Production dependencies
./setup.sh --dev    # Production + development (pytest, ruff)
```

### Option 2: Manual with uv

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv .venv
uv pip install -e ".[dev]"

# Copy environment config
cp .env.example .env
```

### Option 3: Traditional pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI

```bash
uv run vtk-rag chunk              # Process raw data into chunks
uv run vtk-rag index              # Build Qdrant indexes
uv run vtk-rag build              # Full pipeline (chunk + index)
uv run vtk-rag clean              # Remove processed data and indexes
uv run vtk-rag search "query"     # Search code and docs
```

Or activate the venv and run directly:
```bash
source .venv/bin/activate
vtk-rag build
```

### Search Options

```bash
vtk-rag search "query" -n 10           # Limit results
vtk-rag search "query" --hybrid        # Hybrid search (dense + BM25)
vtk-rag search "query" --bm25          # BM25 keyword search
vtk-rag search "query" --code          # Code chunks only
vtk-rag search "query" --docs          # Doc chunks only
vtk-rag search "query" --role source_geometric  # Filter by role
vtk-rag search "query" -v              # Verbose (show content)
```

## Python API

```python
from vtk_rag.config import get_config
from vtk_rag.rag import RAGClient
from vtk_rag.retrieval import Retriever

config = get_config()
rag_client = RAGClient(config)
retriever = Retriever(rag_client)

# Semantic search
results = retriever.search("create a sphere", collection="vtk_code")

# Hybrid search (dense + BM25)
results = retriever.hybrid_search("vtkSphereSource", collection="vtk_docs")

# BM25 keyword search
results = retriever.bm25_search("vtkConeSource SetRadius")

# Filtered search
results = retriever.search(
    "render pipeline",
    filters={"role": "source_geometric", "visibility_score": {"gte": 0.7}},
)

# Convenience methods
results = retriever.search_code("how to create a cylinder")
results = retriever.search_docs("vtkPolyDataMapper")
results = retriever.search_by_class("vtkSphereSource")

# Access results
for r in results:
    print(f"{r.class_name}: {r.synopsis}")
    print(f"  Score: {r.score:.3f}")
    print(f"  Code:\n{r.content}")
```

## Testing

```bash
uv run pytest tests/              # Run tests
uv run pytest tests/ -v           # Verbose output
uv run ruff check vtk_rag/ tests/ # Lint code
```

---

# Architecture

## Data Flow

```
Raw Data (JSONL)                    
    │                               
    ▼                               
┌─────────┐     ┌─────────┐        
│ Chunker │────→│ Indexer │        
└─────────┘     └─────────┘        
    │               │              
    ▼               ▼              
code-chunks.jsonl   Qdrant         
doc-chunks.jsonl    Collections    
                        │          
                        ▼          
                  ┌───────────┐    
                  │ Retriever │    
                  └───────────┘    
                        │          
                        ▼          
                  SearchResults    
```

## Collections

| Collection | Chunks | Description |
|------------|--------|-------------|
| `vtk_code` | ~17,300 | Code examples from VTK examples/tests |
| `vtk_docs` | ~52,600 | Class/method documentation |

---

# Chunking Module

Semantic chunking for VTK Python code and class/method documentation.

## Code Chunks

### Chunk Types

| Type | Description |
|------|-------------|
| **Visualization Pipeline** | property→mapper→actor groups |
| **Rendering Infrastructure** | camera/lights→renderer→window→interactor groups |
| **vtkmodules.{module}** | sources, filters, readers, writers (individual chunks) |

### Semantic Grouping

The `LifecycleAnalyzer` tracks VTK object lifecycles and groups them:

1. **Visualization pipelines** - Groups property + mapper + actor via `SetMapper`/`SetProperty`
2. **Rendering infrastructure** - Combines cameras, lights, renderers, windows, interactors
3. **Query elements** - Sources, readers, writers, filters as individual chunks

### Code Chunk Metadata

| Field | Description |
|-------|-------------|
| `chunk_id` | Unique identifier |
| `example_id` | Source example URL |
| `type` | Chunk type |
| `function_name` | Containing function |
| `title` | Human-readable title |
| `description` | Detailed description |
| `synopsis` | Natural language summary |
| `content` | Executable Python code with imports |
| `roles` | Functional roles (source_geometric, mapper_polydata, etc.) |
| `visibility_score` | User-facing likelihood (0.0-1.0) |
| `input_datatype` / `output_datatype` | Data types |
| `vtk_class` | Primary VTK class |
| `queries` | Pre-generated search queries |

## Doc Chunks

### Chunk Types

| Type | Description |
|------|-------------|
| **class_overview** | Class description and synopsis |
| **constructor** | How to instantiate the class |
| **property_group** | Related Set/Get/On/Off methods grouped by property |
| **standalone_methods** | Methods not part of property groups |
| **inheritance** | Parent class hierarchy |

### Property Grouping

VTK methods are grouped by property name:
- `SetRadius`, `GetRadius`, `GetRadiusMinValue`, `GetRadiusMaxValue` → one chunk
- `ScalarVisibilityOn`, `ScalarVisibilityOff`, `SetScalarVisibility`, `GetScalarVisibility` → one chunk

### Doc Chunk Metadata

| Field | Description |
|-------|-------------|
| `chunk_id` | Unique identifier |
| `chunk_type` | Type (class_overview, constructor, property_group, etc.) |
| `class_name` | VTK class name |
| `content` | Full documentation text |
| `synopsis` | Brief summary |
| `role` | Functional role |
| `action_phrase` | Concise action description |
| `visibility` | User-facing likelihood |
| `queries` | Pre-generated search queries |

## Query Generation

Queries are pre-generated for each chunk to improve search recall:

**Code chunks:** Pattern templates, configuration categories, synopsis values  
**Doc chunks:** Action phrases, camelCase→words conversion, class names

---

# Indexing Module

Index VTK chunks into Qdrant for hybrid search.

## Search Architecture

Each collection supports three search modes:

### Dense Vectors (Semantic)
- **Model**: SentenceTransformer (`all-MiniLM-L6-v2`, 384-dim)
- **content**: Single embedding of chunk text
- **queries**: Multi-vector of pre-generated query embeddings

### Sparse Vectors (BM25)
- **Model**: FastEmbed (`Qdrant/bm25`)
- **bm25**: Sparse embedding with IDF weighting
- Good for exact VTK class names like `vtkSphereSource`

### Payload Indexes (Filtering)
- **keyword**: Exact match (fast)
- **text**: Tokenized full-text
- **float**: Range queries

### Hybrid Search
Combine dense + sparse with Reciprocal Rank Fusion (RRF).

## Collection Fields

### vtk_code

| Field | Type | Description |
|-------|------|-------------|
| `content` | Dense | Semantic similarity |
| `bm25` | Sparse | BM25 keyword matching |
| `queries` | Multi | Pre-generated queries |
| `type` | Keyword | Visualization Pipeline, Rendering Infrastructure, vtkmodules.* |
| `vtk_class` | Keyword | Primary VTK class |
| `function_name` | Keyword | Containing function |
| `roles` | Keyword | Functional roles |
| `input_datatype` | Keyword | Input data type |
| `output_datatype` | Keyword | Output data type |
| `visibility_score` | Float | User-facing likelihood (0.0-1.0) |

### vtk_docs

| Field | Type | Description |
|-------|------|-------------|
| `content` | Dense | Semantic similarity |
| `bm25` | Sparse | BM25 keyword matching |
| `queries` | Multi | Pre-generated queries |
| `chunk_type` | Keyword | class_overview, constructor, property_group, etc. |
| `class_name` | Keyword | VTK class name |
| `role` | Keyword | Functional role |
| `visibility` | Keyword | User-facing likelihood |
| `metadata.module` | Keyword | VTK module path |

---

# Retrieval Module

Core retrieval primitives for searching VTK code and documentation.

## Search Modes

### Semantic Search
Dense vector similarity using SentenceTransformer embeddings.
Best for natural language queries.

```python
results = retriever.search("how do I visualize medical imaging data")
```

### BM25 Search
Sparse vector keyword matching using FastEmbed BM25.
Best for exact VTK class/method names.

```python
results = retriever.bm25_search("vtkDICOMImageReader")
```

### Hybrid Search
Combines dense + sparse with Reciprocal Rank Fusion (RRF).
Best for mixed queries with both natural language and VTK terms.

```python
results = retriever.hybrid_search("create sphere using vtkSphereSource")
```

### Multi-Vector Search
Search against pre-generated query embeddings for better recall.

```python
results = retriever.search("sphere", vector_name="queries")
```

## Filtering

Filters narrow search results by metadata fields. No Qdrant imports required.

### Dict Syntax (Simple)

For basic filters with AND logic only:

```python
# Exact match
results = retriever.search("sphere", filters={"role": "source_geometric"})

# Match any
results = retriever.search("sphere", filters={
    "class_name": ["vtkSphereSource", "vtkConeSource"]
})

# Range
results = retriever.search("sphere", filters={
    "visibility_score": {"gte": 0.7}
})

# Combined (all must match)
results = retriever.search("sphere", filters={
    "type": "Visualization Pipeline",
    "visibility_score": {"gte": 0.5},
})
```

### FilterBuilder (Full Control)

For exclusions, optional matches, or complex logic:

```python
from vtk_rag.retrieval import FilterBuilder

filters = (
    FilterBuilder()
    .match("role", "source_geometric")           # must match exactly
    .match_any("vtk_class", ["vtkSphereSource", "vtkConeSource"])  # must match one
    .range("visibility_score", gte=0.7)          # must be >= 0.7
    .exclude("chunk_type", "inheritance")        # must NOT match
    .should_match("type", "Visualization Pipeline")  # bonus if matches
    .build()
)

results = retriever.search("sphere", filters=filters)
```

### Available Filter Fields

**Code collection (vtk_code):**
- `type`, `vtk_class`, `function_name`, `roles`
- `input_datatype`, `output_datatype`, `visibility_score`
- `example_id`, `variable_name`

**Doc collection (vtk_docs):**
- `chunk_type`, `class_name`, `role`, `visibility`
- `metadata.module`, `metadata.input_datatype`, `metadata.output_datatype`

## SearchResult

```python
result = results[0]

# Core fields
result.id          # Qdrant point ID
result.score       # Relevance score
result.content     # Chunk text
result.chunk_id    # Original chunk identifier
result.collection  # vtk_code or vtk_docs
result.payload     # Full metadata dict

# Common properties
result.class_name       # VTK class name
result.chunk_type       # Chunk type
result.synopsis         # Brief summary
result.role             # Primary functional role
result.input_datatype   # Input data type
result.output_datatype  # Output data type
result.module           # VTK module path

# Code chunk properties
result.title            # Human-readable title
result.description      # Detailed description
result.example_id       # Source example URL
result.function_name    # Containing function
result.variable_name    # Primary variable
result.roles            # All functional roles (list)
result.visibility_score # User-facing likelihood (0.0-1.0)

# Doc chunk properties
result.action_phrase    # Concise action description
```

---

# Code Map

```
vtk_rag/
├── __init__.py
├── __main__.py          # python -m vtk_rag
├── cli.py               # Unified CLI
├── build.py             # Build pipeline
├── config.py            # Configuration management
│
├── chunking/
│   ├── __init__.py      # Exports: Chunker, CodeChunker, DocChunker
│   ├── chunker.py       # Chunker orchestrator
│   │
│   ├── code/            # Code chunking
│   │   ├── __init__.py
│   │   ├── chunker.py       # CodeChunker class
│   │   ├── models.py        # CodeChunk dataclass
│   │   ├── semantic_chunk.py # SemanticChunk builder
│   │   │
│   │   └── lifecycle/       # VTK lifecycle analysis
│   │       ├── __init__.py
│   │       ├── analyzer.py      # LifecycleAnalyzer
│   │       ├── visitor.py       # AST visitor for VTK patterns
│   │       ├── builder.py       # Build lifecycles from context
│   │       ├── grouper.py       # Group lifecycles semantically
│   │       ├── models.py        # LifecycleContext, VTKLifecycle
│   │       ├── utils.py         # Deduplication helpers
│   │       └── vtk_knowledge.py # VTK class/method patterns
│   │
│   ├── doc/             # Doc chunking
│   │   ├── __init__.py
│   │   ├── chunker.py       # DocChunker class
│   │   └── models.py        # DocChunk dataclass
│   │
│   └── query/           # Query generation
│       ├── __init__.py
│       └── semantic_query.py # SemanticQuery builder
│
├── mcp/
│   ├── __init__.py      # Exports: VTKClient, get_vtk_client
│   └── client.py        # VTK API client
│
├── rag/
│   ├── __init__.py      # Exports: RAGClient
│   └── client.py        # RAGClient (Qdrant + embeddings)
│
├── indexing/
│   ├── __init__.py      # Exports: Indexer
│   ├── indexer.py       # Indexer class
│   └── models.py        # CollectionConfig, FieldConfig
│
└── retrieval/
    ├── __init__.py      # Exports: Retriever, SearchResult
    ├── retriever.py     # Retriever class
    └── models.py        # SearchResult dataclass

tests/
├── conftest.py              # Fixtures
├── test_chunk_quality.py    # Chunk quality validation
├── test_chunking.py         # Chunker tests
├── test_cli.py              # CLI tests
├── test_indexing.py         # Indexer tests
├── test_lifecycle.py        # Lifecycle analysis tests
├── test_rag_client.py       # RAGClient tests
├── test_rendering_chunks.py # Rendering chunk tests
└── test_retrieval.py        # Retriever tests

examples/
├── search_examples.py       # Search API demo
└── show_chunks_by_role.py   # Display chunks by role
```

---

# Prerequisites

- **Python 3.10+**
- **Qdrant**: `docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant`
- **Raw data files** in `data/raw/`:
  - `vtk-python-docs.jsonl` (~2,900 classes)
  - `vtk-python-examples.jsonl` (~850 examples)
  - `vtk-python-tests.jsonl` (~900 tests)

---

# License

MIT License
