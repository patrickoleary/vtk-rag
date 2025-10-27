# VTK Corpus Preparation - Content-Type Separation

Prepares VTK Python documentation for RAG by chunking into **separate content types** for targeted retrieval and optimal token usage.

## Overview

Processes VTK documentation into **4 distinct chunk types**:

1. **CODE chunks** - Pure executable code with imports
2. **EXPLANATION chunks** - Descriptions, tutorials, concepts
3. **API_DOC chunks** - Method documentation
4. **IMAGE chunks** - Image metadata and links

**Input:**
- `vtk-python-examples.jsonl` (~847 examples)
- `vtk-python-tests.jsonl` (~909 tests)
- `vtk-python-docs.jsonl` (~2,942 API classes)

**Output:**
- `code_chunks.jsonl` (~3,594 chunks)
- `explanation_chunks.jsonl` (~1,479 chunks)
- `api_doc_chunks.jsonl` (~125,708 chunks)
- `image_chunks.jsonl` (~281 chunks)

**Total:** ~1,756 documents → **131,062 chunks**

---

## Quick Usage

```bash
# Process all files (Phase 7)
python prepare-corpus/chunk_corpus.py

# View statistics
cat data/processed/statistics.json | python3 -m json.tool

# Try example usage
python prepare-corpus/example_usage.py
```

**Runtime:** ~2-3 minutes for all files

---

## File Inventory

### Core Scripts (Used in Pipeline)

| File | Purpose | Used By |
|------|---------|---------|
| `chunk_corpus.py` | Main chunking orchestrator | Pipeline entry point |
| `code_chunker.py` | Code analysis & metadata detection | chunk_corpus.py |
| `explanation_chunker.py` | Explanation text chunking | chunk_corpus.py |
| `image_chunker.py` | Image metadata extraction | chunk_corpus.py |
| `api_doc_chunker.py` | API documentation chunking | chunk_corpus.py |
| `query_generator.py` | Generate search queries for chunks | chunk_corpus.py |

### Utility Scripts

| File | Purpose | Notes |
|------|---------|-------|
| `example_usage.py` | Usage examples and filtering patterns | ✅ Tested and working |
| `analyze_corpus.py` | Corpus statistics and analysis | Standalone utility |

### Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | Dependencies (matplotlib optional) |
| `README.md` | This documentation |

---

## Content-Type Separation Architecture

### Why Separate by Content Type?

Content is separated into distinct types (code, explanation, API docs, images) to enable:

- **Targeted retrieval:** Each query retrieves only the content type it needs
- **Token efficiency:** Smaller, focused chunks (~400t for code vs 1500t for mixed content)
- **Quality filtering:** Metadata enables filtering by API style, import pattern, and complexity
- **Flexible composition:** Combine different content types as needed per query

**Typical usage:** 3 code chunks × 400t = 1,200 tokens per query (vs 10,500t with mixed content)

### Chunking Strategies by Type

**Common fields for all chunks:**
- `chunk_id` - Unique identifier
- `chunk_index` - Position in multi-chunk sequence (0-indexed)
- `total_chunks` - Total chunks for this document
- `content` - The actual text content
- `content_type` - `code`, `explanation`, `api_doc`, or `image`
- `source_type` - `example`, `test`, or `api`
- `original_id` - Source document identifier

---

#### 1. CODE Chunks (~400 tokens)

**Pure executable code with rich metadata**

```json
{
  "content_type": "code",
  "content": "from vtkmodules.vtkFiltersSources import vtkCylinderSource\n...",
  "metadata": {
    "source_style": "pythonic",        // "pythonic", "basic", or "snippet"
    "import_style": "modular",         // "modular", "monolithic", "mixed", or "none"
    "vtk_classes": ["vtkCylinderSource", "vtkActor", ...],
    "has_visualization": true,
    "has_data_io": false,
    "requires_data_files": false,
    "complexity": "simple",             // "simple", "moderate", or "complex"
    "data_files": [],
    "data_download_info": []
  }
}
```

**Metadata Fields:**

- `source_style` - API syntax style:
  - `"pythonic"` - Modern API (properties, `>>`, keyword args) **[981 examples, 27.3%]**
  - `"basic"` - Traditional API (Set/Get methods) **[2,610 examples, 72.6%]**
  - `"snippet"` - Helper functions without VTK imports **[3 examples, 0.1%]**
- `import_style` - Import pattern style:
  - `"modular"` - Selective imports: `from vtkmodules.vtkX import Y` **[3,541 examples, 98.5%]**
  - `"monolithic"` - Full import: `import vtk` or `from vtkmodules.all` **[2 examples, 0.1%]**
  - `"mixed"` - Both patterns (bad practice) **[0 examples - clean corpus!]**
  - `"none"` - No VTK imports (snippets/utilities) **[51 examples, 1.4%]**
- `vtk_classes` - List of VTK classes used
- `has_visualization` - Boolean (has rendering/display code)
- `has_data_io` - Boolean (reads/writes files)
- `has_filters` - Boolean (uses VTK filters)
- `has_sources` - Boolean (uses VTK data sources)
- `has_mappers` - Boolean (uses VTK mappers)
- `requires_data_files` - Boolean (needs external data files)
- `data_files` - List of required files (if any)
- `data_download_info` - Download URLs and hashes
- `complexity` - `"simple"`, `"moderate"`, or `"complex"`
- `line_count` - Number of lines
- `function_count` - Number of functions
- `related_explanation_chunk` - Link to explanation chunk (if exists)
- `related_image_chunk` - Link to image chunk (if exists)

**Key Features:**
- ✅ **Gold Standard:** 971 examples with pythonic API + modular imports (27%)
- ✅ **Quality Boosting:** Pythonic +20%, modular +15% in retrieval
- ✅ Self-contained detection (no external data needed)
- ✅ Full import statements included

#### 2. EXPLANATION Chunks (~500 tokens)

**Descriptions, tutorials, concepts**

```json
{
  "content_type": "explanation",
  "content": "# CylinderExample\n**Category:** GeometricObjects\n\nThis example demonstrates...",
  "metadata": {
    "title": "CylinderExample",
    "category": "GeometricObjects",
    "has_description": true,
    "related_code_chunk": "CylinderExample_code_0",
    "related_image_chunk": "CylinderExample_image_0"
  }
}
```

**Metadata Fields:**

- `title` - Example/test title
- `category` - Category (for examples, e.g., "GeometricObjects", "Rendering")
- `has_description` - Boolean (has descriptive text)
- `related_code_chunk` - Link to code chunk (if exists)
- `related_image_chunk` - Link to image chunk (if exists)

**Key Features:**
- ✅ Links to related code and image chunks
- ✅ Category-based filtering
- ✅ Preserves markdown structure

#### 3. API_DOC Chunks (~300 tokens)

**Method documentation with class context**

```json
{
  "content_type": "api_doc",
  "content": "# vtkActor\n**Module:** vtkRenderingCore\n\n## AddPosition\n...",
  "metadata": {
    "class_name": "vtkActor",
    "module_name": "vtkmodules.vtkRenderingCore",
    "method_names": ["AddPosition", "GetProperty"],
    "method_count": 2
  }
}
```

**Metadata Fields:**

- `class_name` - VTK class name (e.g., "vtkActor")
- `module_name` - Python module (e.g., "vtkmodules.vtkRenderingCore")
- `method_names` - List of methods in this chunk
- `method_count` - Number of methods

**Key Features:**
- ✅ Class context in every chunk
- ✅ Multiple methods per chunk for efficiency
- ✅ Module information for correct imports

#### 4. IMAGE Chunks (~50 tokens)

**Image metadata with links**

```json
{
  "content_type": "image",
  "content": "![CylinderExample](https://examples.vtk.org/.../CylinderExample.png)\n\n**Example:** CylinderExample",
  "metadata": {
    "image_url": "https://...",
    "image_title": "CylinderExample",
    "image_type": "result",           // or "baseline"
    "related_code_chunk": "CylinderExample_code_0"
  }
}
```

**Metadata Fields:**

- `image_url` - URL to image
- `image_title` - Display title
- `image_type` - `"result"` or `"baseline"`
- `related_code_chunk` - Link to code chunk (if exists)

**Key Features:**
- ✅ Links to code chunks
- ✅ Result images vs baseline images
- ✅ Minimal tokens for efficient multimodal RAG

---

## Retrieval Patterns

### Code Query (Pythonic, Self-Contained)

```python
filter = {
    "must": [
        {"key": "content_type", "match": {"value": "code"}},
        {"key": "metadata.source_style", "match": {"value": "pythonic"}},
        {"key": "metadata.requires_data_files", "match": {"value": False}}
    ]
}
# Returns: 3 chunks × ~400t = 1,200 tokens
```

### Explanation Query

```python
filter = {
    "must": [
        {"key": "content_type", "match": {"value": "explanation"}},
        {"key": "metadata.category", "match": {"value": "Rendering"}}
    ]
}
# Returns: 5 chunks × ~500t = 2,500 tokens
```

### API Query

```python
filter = {
    "must": [
        {"key": "content_type", "match": {"value": "api_doc"}},
        {"key": "metadata.class_name", "match": {"value": "vtkActor"}}
    ]
}
# Returns: Focused API documentation
```

### Image Query

```python
filter = {
    "must": [
        {"key": "content_type", "match": {"value": "image"}},
        {"key": "metadata.image_type", "match": {"value": "result"}}
    ]
}
# Returns: Result images with metadata (~50t each)
```

---

## Output Files

### Directory Structure

```
data/processed/
├── code_chunks.jsonl          (3,594 chunks)
├── explanation_chunks.jsonl   (1,479 chunks)
├── api_doc_chunks.jsonl       (125,708 chunks)
├── image_chunks.jsonl         (281 chunks)
└── statistics.json            (summary stats)
```

### Statistics Output

```json
{
  "total_chunks": 131062,
  "by_content_type": {
    "code": 3594,
    "explanation": 1479,
    "api_doc": 125708,
    "image": 281
  },
  "by_source_type": {
    "example": 4854,
    "test": 500,
    "api": 125708
  }
}
```

---

## Chunking Implementation

### Main Script

**`chunk_corpus.py`** - Content-type aware chunking

```bash
python prepare-corpus/chunk_corpus.py
```

**Features:**
- Separate chunkers for each content type
- Code analysis (detect pythonic style, complexity)
- Metadata enrichment
- Link creation between related chunks
- Statistics generation

### Chunker Classes

1. **`CodeOnlyChunker`** (`code_chunker.py`)
   - Extracts executable code
   - Detects API style (pythonic/basic/snippet)
   - Detects import pattern (modular/monolithic/mixed/none)
   - Analyzes complexity (simple/moderate/complex)
   - Detects required data files
   - Identifies VTK classes used
   - **Backend import handling** (OpenGL2, FreeType, InteractionStyle)

2. **`ExplanationChunker`**
   - Extracts descriptions
   - Creates links to code/images
   - Preserves markdown structure

3. **`ApiDocChunker`**
   - Splits by methods
   - Preserves class context
   - Groups methods efficiently

4. **`ImageChunker`**
   - Creates minimal metadata chunks
   - Links to related code
   - Preserves image URLs

---

## Example Usage

### Load and Filter Chunks

```python
from pathlib import Path
import json

# Load CODE chunks
with open('data/processed/code_chunks.jsonl') as f:
    code_chunks = [json.loads(line) for line in f]

# Filter pythonic, self-contained examples
pythonic = [
    c for c in code_chunks 
    if c['metadata']['source_style'] == 'pythonic'
    and not c['metadata']['requires_data_files']
]

print(f"Pythonic, self-contained examples: {len(pythonic)}")
```

See **`example_usage.py`** for more examples.

---

## Performance

- **Processing Speed:** ~500-800 docs/sec
- **Memory Usage:** ~200-400 MB (streaming)
- **Runtime:** 2-3 minutes for all files
- **Output Size:** ~138 MB total

---

## Design Rationale

### Content-Type Separation

Chunks are separated by content type (code/explanation/api_doc/image) to enable:

- **Targeted retrieval:** Retrieve only the content type needed for each query
- **Token efficiency:** Code chunks (~400t) are smaller than mixed chunks, reducing LLM context size
- **Semantic focus:** Pure code examples without explanatory text clutter
- **Flexible composition:** Combine different content types as needed per query

**Result:** Typical queries use ~1,200 tokens (3 code chunks) vs 10,500 tokens with mixed content.

### Rich Metadata for Quality-Aware Retrieval

Each code chunk includes metadata to enable intelligent filtering:

- **API Style** (`source_style`): Distinguish pythonic (modern) vs basic (traditional) API usage
- **Import Pattern** (`import_style`): Identify modular imports vs monolithic imports
- **Self-Contained** (`requires_data_files`): Filter examples that don't need external files
- **Complexity** (`complexity`): Match simple, moderate, or complex examples to query difficulty

**Result:** Retrieval systems can prioritize the 971 gold-standard examples (pythonic + modular) and exclude examples requiring data files.

### Cross-Linking for Context

Related chunks are linked via metadata:

- Code chunks link to explanation and image chunks
- Explanation chunks link to code and images
- Image chunks link back to code

**Result:** Retrieve a code example and optionally fetch its explanation or result image on demand.

### Multimodal Support

Image chunks are separated to minimize token usage while enabling visual context:

- **Lightweight metadata:** ~50 tokens per image (vs embedding full images)
- **On-demand retrieval:** Only include images when needed
- **Result preview:** Link to visual output of code examples

---

## Next Steps

After chunking, build the vector database index:

```bash
python build-indexes/build_qdrant_index.py
```

This creates embeddings for all chunks and indexes them in Qdrant for semantic search.
