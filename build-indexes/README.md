# Build Qdrant Vector Index - Content-Type Separation

Build a vector database with **content-type filtering** from VTK chunked corpus using Qdrant.

## Overview

Builds a Qdrant vector database from chunked VTK corpus with:

- **Content-type separation** - Index code, explanations, API docs, and images separately
- **Rich metadata filtering** - Filter by API style, import pattern, complexity, data requirements
- **Quality-aware retrieval** - Support for prioritizing pythonic examples (971 gold-standard)
- **Token efficiency** - Typical queries use ~1,200 tokens vs 10,500 with mixed content
- **Production-ready** - Fast, scalable, self-hosted

## Why Qdrant?

- ✅ **Metadata filtering** - Filter by content_type, source_style, complexity
- ✅ **Easy setup** - One Docker command
- ✅ **FREE & open source** - Self-hosted
- ✅ **Fast** - Optimized in Rust
- ✅ **Production-ready** - Great scaling
- ✅ **Built-in BM25** - Hybrid search support

---

## Quick Start

### 1. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Open http://localhost:6333/dashboard to see the web UI.

### 2. Install Dependencies

```bash
pip install -r build-indexes/requirements.txt
```

Installs:
- `qdrant-client` - Qdrant Python client
- `sentence-transformers` - Embedding model

### 3. Build Index (Phase 8)

```bash
# Index all chunks (~131,000 chunks, ~3-5 minutes)
python build-indexes/build_qdrant_index.py
```

### 4. Verify Index

```bash
# Run example queries to verify the index works
python build-indexes/example_usage.py
```

This demonstrates:
- Filtered search by content_type
- Pythonic code filtering
- Self-contained examples
- API class lookup
- Complexity filtering

---

## File Inventory

### Core Scripts (Used in Pipeline)

| File | Purpose | Used By |
|------|---------|------|
| `build_qdrant_index.py` | Build Qdrant vector index from processed chunks | Pipeline entry point |

### Utility Scripts

| File | Purpose | Notes |
|------|---------|-------|
| `example_usage.py` | Demonstrate query patterns and filtering | ✅ Tested and working |

### Configuration

| File | Purpose |
|------|------|
| `requirements.txt` | Dependencies (qdrant-client, sentence-transformers) |
| `README.md` | This documentation |

---

## Index Structure

### Total: 131,062 Chunks

**By Content Type:**
- **CODE**: 3,594 chunks (~400 tokens avg)
- **EXPLANATION**: 1,479 chunks (~91 tokens avg)
- **API_DOC**: 125,708 chunks (~139 tokens avg)
- **IMAGE**: 281 chunks (metadata only)

**By Source:**
- Examples: 5,854 chunks
- Tests: 500 chunks
- API Docs: 125,708 chunks

### Metadata Indexes Created

The index includes filters on:
- `content_type` (code/explanation/api_doc/image)
- `source_type` (example/test/api)
- `metadata.source_style` (pythonic/basic/snippet)
- `metadata.import_style` (modular/monolithic/mixed/none)
- `metadata.has_visualization` (bool)
- `metadata.has_data_io` (bool)
- `metadata.requires_data_files` (bool)
- `metadata.complexity` (simple/moderate/complex)

**Quality Metadata:**
- **971 gold-standard examples** (pythonic API + modular imports)
- Enables quality-aware retrieval with score boosting
- See `retrieval-pipeline/task_specific_retriever.py` for usage

---

## Performance

**Indexing Time** (~3-5 minutes on modern hardware):
- CODE chunks (3,594): ~30 seconds
- EXPLANATION chunks (1,479): ~15 seconds
- API_DOC chunks (125,708): ~3-4 minutes
- IMAGE chunks (281): ~5 seconds

**Memory Usage**: ~500MB during indexing

**Index Size**: ~1.2GB on disk

---

## Retrieval Patterns

### 1. Code Query (Pythonic, Self-Contained)

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer('all-MiniLM-L6-v2')

query = "How do I create a cylinder in VTK?"
query_vector = model.encode(query)

results = client.search(
    collection_name="vtk_docs",
    query_vector=query_vector.tolist(),
    query_filter={
        "must": [
            {"key": "content_type", "match": {"value": "code"}},
            {"key": "metadata.source_style", "match": {"value": "pythonic"}},
            {"key": "metadata.requires_data_files", "match": {"value": False}}
        ]
    },
    limit=3
)

# Returns: ~3 chunks × 400t = 1,200 tokens
```

### 2. Explanation Query

```python
results = client.search(
    collection_name="vtk_docs",
    query_vector=query_vector.tolist(),
    query_filter={
        "must": [
            {"key": "content_type", "match": {"value": "explanation"}},
        ]
    },
    limit=5
)

# Returns: ~5 chunks × 91t = 455 tokens
```

### 3. API Documentation Query

```python
results = client.search(
    collection_name="vtk_docs",
    query_vector=query_vector.tolist(),
    query_filter={
        "must": [
            {"key": "content_type", "match": {"value": "api_doc"}},
            {"key": "metadata.class_name", "match": {"value": "vtkActor"}}
        ]
    },
    limit=3
)

# Returns: Focused documentation for vtkActor class
```

### 4. Image Query

```python
results = client.search(
    collection_name="vtk_docs",
    query_vector=query_vector.tolist(),
    query_filter={
        "must": [
            {"key": "content_type", "match": {"value": "image"}},
            {"key": "metadata.image_type", "match": {"value": "result"}}
        ]
    },
    limit=3
)

# Returns: Result images with metadata (~0 tokens, lightweight)
```

### 5. Combined Retrieval (Code + Explanation + Image)

```python
# Get code
code_results = client.search(
    collection_name="vtk_docs",
    query_vector=query_vector.tolist(),
    query_filter={"must": [{"key": "content_type", "match": {"value": "code"}}]},
    limit=1
)

# Get related explanation using chunk links
code_chunk = code_results[0].payload
related_explanation_id = code_chunk['metadata'].get('related_explanation_chunk')

# Get related image
related_image_id = code_chunk['metadata'].get('related_image_chunk')

# Total tokens: 400 (code) + 91 (explanation) + 0 (image metadata) = ~491 tokens
```

---

## Token Efficiency

**Content-type filtering enables targeted retrieval with minimal tokens:**

**Example: "How to create a cylinder?"**

Retrieve exactly what you need:
- 1 CODE chunk: ~400 tokens (pythonic, self-contained)
- 1 EXPLANATION chunk: ~91 tokens (descriptive context)
- 1 IMAGE chunk: 0 tokens (metadata only, links to visual result)
- **Total: ~491 tokens**

By filtering to specific content types, queries use only the tokens needed for the task—no mixed content, no unnecessary context.

---

## Testing Your Index

After indexing, test with:

```bash
python build-indexes/example_usage.py
```

**Example queries demonstrated:**
- ✅ Code-only retrieval (pythonic style)
- ✅ Explanation-only retrieval
- ✅ API documentation retrieval
- ✅ Image retrieval (result images)
- ✅ Filter by complexity
- ✅ Filter by data file requirements
- ✅ VTK class filtering

---

## Advanced Filtering

### Find Simple, Self-Contained Examples

```python
results = client.search(
    collection_name="vtk_docs",
    query_vector=query_vector.tolist(),
    query_filter={
        "must": [
            {"key": "content_type", "match": {"value": "code"}},
            {"key": "metadata.complexity", "match": {"value": "simple"}},
            {"key": "metadata.requires_data_files", "match": {"value": False}},
            {"key": "metadata.has_visualization", "match": {"value": True}}
        ]
    },
    limit=3
)
```

### Find Examples Using Specific VTK Classes

```python
results = client.search(
    collection_name="vtk_docs",
    query_vector=query_vector.tolist(),
    query_filter={
        "must": [
            {"key": "content_type", "match": {"value": "code"}},
            {"key": "metadata.vtk_classes", "match": {"any": ["vtkActor", "vtkRenderer"]}}
        ]
    },
    limit=5
)
```

---

## Command-Line Options

```bash
# Build index (default: all chunks)
python build-indexes/build_qdrant_index.py

# Use different Qdrant URL
python build-indexes/build_qdrant_index.py --url http://remote-server:6333

# Use different embedding model
python build-indexes/build_qdrant_index.py --model all-mpnet-base-v2

# Custom collection name
python build-indexes/build_qdrant_index.py --collection my_vtk_docs

# Adjust batch size
python build-indexes/build_qdrant_index.py --batch-size 50

# Skip test query
python build-indexes/build_qdrant_index.py --skip-test
```

---

## Output

After indexing completes, you'll have:

- ✅ **131,062 searchable chunks** with vector embeddings
- ✅ **Web UI** at http://localhost:6333/dashboard
- ✅ **Content-type filtering** for targeted retrieval
- ✅ **Rich metadata** for precise filtering
- ✅ **Links between related chunks** (code ↔ explanation ↔ image)
- ✅ **Production-ready** index for RAG applications

---

## Troubleshooting

### Docker container not running
```bash
# Check if running
docker ps | grep qdrant

# Start Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Connection refused
```bash
# Check port availability
lsof -i :6333

# Verify Qdrant is responding
curl http://localhost:6333/
```

### Out of memory during indexing
- Reduce batch size: `--batch-size 50`
- Index is memory-intensive due to embedding generation
- Ensure at least 2GB RAM available

### Slow indexing
- Normal for 125K+ chunks
- API docs are largest (125,708 chunks)
- First run downloads embedding model (~80MB)

---

## Additional Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [Sentence Transformers Models](https://www.sbert.net/docs/pretrained_models.html)
