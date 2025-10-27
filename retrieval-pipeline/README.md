# VTK RAG Retrieval Pipeline - TaskSpecificRetriever

Content-type aware retrieval with quality boosting and token efficiency.

## Overview

TaskSpecificRetriever retrieves targeted chunks by content type (CODE, EXPLANATION, API_DOC, IMAGE) with:

- **Content-type filtering** - Retrieve only the content type needed per query
- **Quality boosting** - Prioritize 971 gold-standard examples (pythonic +20%, modular +15%)
- **Metadata filtering** - Filter by complexity, VTK classes, data requirements
- **Token efficiency** - Typical queries use 400-900 tokens vs 10,500 with mixed content
- **Flexible retrieval** - Single or multiple content types per query

---

## Quick Start

### 1. Ensure Qdrant is Running

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Build Index (if not already done)

```bash
python build-indexes/build_qdrant_index.py
```

### 3. Run Examples

```bash
python retrieval-pipeline/example_usage.py
```

---

## File Inventory

### Core Scripts (Used in Pipeline)

| File | Purpose | Used By |
|------|---------|---------|
| `task_specific_retriever.py` | Task-specific retrieval with quality boosting | Pipeline entry point |

### Utility Scripts

| File | Purpose | Notes |
|------|---------|-------|
| `example_usage.py` | Demonstrate retrieval patterns | Demo script |

### Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | Dependencies (uses build-indexes deps) |
| `README.md` | This documentation |

---

## Basic Usage

### CODE Retrieval (Pythonic, Self-Contained)

```python
from task_specific_retriever import TaskSpecificRetriever

retriever = TaskSpecificRetriever()

# Get pythonic, self-contained code examples
results = retriever.retrieve_code(
    "How to create a cylinder?",
    top_k=3,
    prefer_pythonic=True,
    prefer_self_contained=True
)

# Results: ~900 tokens (3 pythonic code chunks)
```

### EXPLANATION Retrieval

```python
# Get descriptions and tutorials
results = retriever.retrieve_explanation(
    "What is a cylinder source?",
    top_k=3,
    category="GeometricObjects"
)

# Results: ~400 tokens (3 explanation chunks)
```

### API Documentation Retrieval

```python
# Get specific class documentation
results = retriever.retrieve_api_doc(
    "vtkActor methods",
    top_k=3,
    class_name="vtkActor"
)

# Results: ~200 tokens (3 API doc chunks)
```

### IMAGE Retrieval

```python
# Get result images (metadata only)
results = retriever.retrieve_image(
    "cylinder visualization output",
    top_k=3,
    image_type="result"
)

# Results: ~0 tokens (3 image metadata entries with links to visuals)
```

### Mixed Retrieval

```python
# Get multiple content types
results = retriever.retrieve_mixed(
    "cylinder example",
    code_k=2,
    explanation_k=2,
    api_k=1
)

# Returns dict: {'code': [...], 'explanation': [...], 'api_doc': [...]}
```

---

## Key Features

### Content-Type Filtering
- **CODE**: Pure executable code with imports (~400 tokens)
- **EXPLANATION**: Descriptions and tutorials (~91 tokens)
- **API_DOC**: Class/method documentation (~139 tokens)
- **IMAGE**: Visual result metadata (minimal)

### Metadata Filtering
- `source_style`: pythonic vs basic
- `requires_data_files`: self-contained vs needs data
- `complexity`: simple, moderate, complex
- `has_visualization`: rendering examples
- `vtk_classes`: filter by VTK classes used
- `category`: filter explanations by topic

### Retrieval Methods

**TaskSpecificRetriever:**
- `retrieve_code()` - CODE chunks with rich filters
- `retrieve_explanation()` - EXPLANATION chunks
- `retrieve_api_doc()` - API_DOC chunks
- `retrieve_image()` - IMAGE chunks (metadata)
- `retrieve_mixed()` - Multiple content types
- `retrieve_with_config()` - Configuration-based retrieval

---

## Examples

### 1. Simple Code Query

```python
results = retriever.retrieve_code("create cylinder", top_k=3)
# ~900 tokens (3 code chunks)
```

### 2. Filter by Complexity

```python
results = retriever.retrieve_code(
    "basic rendering",
    complexity_level="simple",
    require_visualization=True
)
```

### 3. Filter by VTK Classes

```python
results = retriever.retrieve_code(
    "rendering example",
    vtk_classes=["vtkRenderer", "vtkActor"]
)
```

### 4. Self-Contained Only

```python
results = retriever.retrieve_code(
    "visualization example",
    prefer_self_contained=True  # No external data files
)
```

### 5. Image Results

```python
results = retriever.retrieve_image(
    "cylinder visualization output",
    top_k=3,
    image_type="result"  # "result" or "diagram"
)
# Returns lightweight metadata with links to visual results
```

### 6. Examples WITH Data Files (for I/O tutorials)

```python
results = retriever.retrieve_code(
    "reading image files",
    prefer_self_contained=False  # Allow data files
)
```

### 7. Configuration-Based

```python
from task_specific_retriever import TaskType, RetrievalConfig

config = RetrievalConfig(
    task_type=TaskType.CODE_GENERATION,
    prefer_pythonic=True,
    prefer_self_contained=True,
    complexity_level="simple"
)

results = retriever.retrieve_with_config("sphere example", config)
```

---

## Token Efficiency Examples

**Content-type filtering enables minimal token usage:**

| Query Type | Typical Token Usage |
|------------|---------------------|
| Simple code query | ~900 tokens (3 code chunks) |
| Explanation query | ~400 tokens (3 explanation chunks) |
| API query | ~200 tokens (3 API doc chunks) |
| Image query | ~0 tokens (3 image metadata entries) |
| Mixed query | ~900 tokens (2 code + 2 explanation + 1 API) |

By filtering to specific content types, queries retrieve only what's needed for the task.

---

## API Reference

### TaskSpecificRetriever

**Main retrieval class**

**Methods:**
- `retrieve_code(query, top_k, prefer_pythonic, prefer_self_contained, require_visualization, complexity_level, vtk_classes)` - CODE chunks with filters
- `retrieve_explanation(query, top_k, category)` - EXPLANATION chunks
- `retrieve_api_doc(query, top_k, class_name)` - API_DOC chunks
- `retrieve_image(query, top_k, image_type)` - IMAGE chunks (metadata)
- `retrieve_mixed(query, code_k, explanation_k, api_k)` - Multiple content types
- `retrieve_with_config(query, config, top_k)` - Config-based retrieval
- `estimate_total_tokens(results)` - Calculate token usage
- `print_results_summary(results, title)` - Display results

### RetrievalConfig

**Configuration for retrieval strategy**

**Fields:**
- `task_type`: TaskType (CODE_GENERATION, EXPLANATION, API_LOOKUP, MIXED)
- `prefer_pythonic`: bool
- `prefer_self_contained`: bool
- `require_visualization`: Optional[bool]
- `complexity_level`: Optional[str] ("simple", "moderate", "complex")
- `vtk_classes`: Optional[List[str]]
- `category`: Optional[str]
- `class_name`: Optional[str]

### RetrievalResult

**Single retrieval result**

**Fields:**
- `chunk_id`: str - Unique chunk identifier
- `content`: str - The chunk content
- `content_type`: str - "code", "explanation", "api_doc", or "image"
- `source_type`: str - "example", "test", or "api_doc"
- `score`: float - Relevance score
- `metadata`: Dict - Chunk metadata
- `estimate_tokens()`: Method to estimate token count

---

## Testing

Run examples:

```bash
# Quick test
python retrieval-pipeline/task_specific_retriever.py

# Full examples (10 scenarios)
python retrieval-pipeline/example_usage.py
```

---

## Installation

No additional dependencies required. This module uses the same dependencies installed from `build-indexes/`:

- `qdrant-client` - Qdrant Python client
- `sentence-transformers` - Embedding model

If you haven't already, install them:
```bash
pip install -r build-indexes/requirements.txt
```

---

## Architecture

```
Query → TaskSpecificRetriever → Content-Type Filter → Metadata Filter → Quality Boost → Results
         ↓                        ↓                     ↓                ↓             ↓
    CODE/EXPLANATION         code/explanation      pythonic/simple   +20%/+15%    ~400t
```

**Features:**
- Content-type filtering reduces tokens by retrieving only needed types
- Metadata filtering ensures relevant results (complexity, self-contained, VTK classes)
- Quality boosting prioritizes 971 gold-standard examples (pythonic + modular)

---

## Retrieval Strategy Summary

Different query types use different retrieval strategies to find relevant documentation:

### **Implemented Strategies** ✅

| Query Type | Primary Method | Filters | Returns |
|------------|---------------|---------|---------|
| **CODE queries** | `retrieve_code()` | pythonic, self_contained, vtk_classes | Code examples |
| **API queries** | `retrieve_api_doc()` | class_name, category | API documentation |
| **EXPLANATION queries** | `retrieve_explanation()` | category, complexity | Conceptual explanations |
| **IMAGE queries** | `retrieve_image()` | image_type (result/diagram) | Image metadata with URLs |
| **MIXED queries** | `retrieve_mixed()` | Multiple content types | Dict with code + explanation + API |

### **Planned Strategies** ⏳

Advanced retrieval patterns for multimodal queries:

**1. Image-to-Code** (User provides image, wants code)
- Strategy: `retrieve_image()` → find similar results → get `related_code_chunk` IDs → `retrieve_by_ids()`
- Use case: "What code produces this visualization?"

**2. Code-to-Image** (User provides code, wants expected output)
- Strategy: `retrieve_code()` with `vtk_classes` → filter `has_baseline=True` → get `related_image_chunk` IDs
- Use case: "Show me what this code renders"

**3. Data-to-Code** (User has data file, wants usage examples)
- Strategy: `retrieve_code()` with `has_data_files=True` → filter by file extension match
- Use case: "I have points.csv, what can I do with it?"

**4. Code-to-Data** (User has code, wants example data)
- Strategy: Extract reader type from code → `retrieve_code()` with matching file extension → return data files
- Use case: "Do you have example STL files for this code?"

**All required metadata already exists in chunks** - these just need new retrieval methods.

---

## Benefits

✅ **Token-efficient retrieval** - 400-900 tokens per query  
✅ **Targeted by content type** - CODE, EXPLANATION, API_DOC, IMAGE  
✅ **Quality boosting** - Prioritize 971 pythonic+modular examples  
✅ **Automatic pythonic preference** for generated code  
✅ **Self-contained by default** - no data file dependencies  
✅ **Flexible filtering** by complexity, VTK classes, category  
✅ **Configurable strategies** per task type  
✅ **Easy integration** with LLM pipelines  
