# VTK RAG: Retrieval-Augmented Generation for VTK Documentation

A production-ready Retrieval-Augmented Generation (RAG) system that transforms natural language queries about VTK (Visualization Toolkit) into validated, executable Python code with explanations and citations.

---

## Purpose

**Problem:** VTK has a large, complex API (2,900+ classes) with extensive documentation scattered across examples, tests, and API references. Developers need:
- Fast code generation for visualization tasks
- Accurate API documentation lookup
- Validated, executable code (not hallucinations)
- Citations to learn from real examples

**Solution:** A RAG system that:
1. **Indexes** ~131,000 chunks of VTK documentation (125K API docs, 3.6K code examples, 1.5K explanations, 300 images)
2. **Retrieves** relevant documentation for any VTK query
3. **Generates** Python code using LLMs with grounded prompts
4. **Validates** code for API correctness and security
5. **Executes** code in Docker sandbox to verify functionality

**Result:** Query `"How do I create a cylinder?"` → Get working VTK code + explanation + citations in seconds.

---

## Architecture

The system implements a **7-stage RAG pipeline** centered around `query.py`:

```
┌─────────────────────────────────────────────────────────────┐
│                        query.py                             │
│              (Unified Query Interface)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │  Sequential Pipeline       │
        │  (Multi-Step Orchestrator) │
        └─────────────┬───────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    v                 v                 v
┌─────────┐    ┌───────────┐    ┌──────────┐
│Retrieval│───→│ Grounding │───→│   LLM    │
│Pipeline │    │ Prompts   │    │Generation│
└─────────┘    └───────────┘    └──────────┘
    │                                  │
    v                                  v
┌─────────┐                    ┌──────────────┐
│ Qdrant  │                    │  Validation  │
│ Vector  │                    │  & Execution │
│  Index  │                    └──────────────┘
└─────────┘
    │
    v
┌──────────────────────────────────────┐
│  Processed Corpus (~131K chunks)     │
│  • API docs (~125,700)               │
│  • Code examples (~3,600)            │
│  • Explanations (~1,500)             │
│  • Images (~300)                     │
└──────────────────────────────────────┘
```

**Data Flow:**
1. **User Query** → `query.py`
2. **Query Decomposition** → Break complex queries into steps
3. **Retrieval** → Find relevant docs for each step (Qdrant)
4. **Grounding** → Build prompts with retrieved context
5. **Generation** → LLM generates code + explanation
6. **Validation** → Check API correctness & security
7. **Execution** (optional) → Run code in Docker sandbox

---

## Repository Structure

Components are organized in the order they're used in the pipeline:

### **1. Data Preparation** (`data/`, `prepare-corpus/`)

**Purpose:** Transform raw VTK documentation into searchable chunks

**Components:**
- `data/raw/` - Input JSONL files (API docs, examples, tests)
- `prepare-corpus/chunk_corpus.py` - Main chunking orchestrator
  - Structure-aware chunking (preserves code blocks, explanations)
  - Metadata extraction (VTK classes, complexity, categories)
  - Deduplication and quality filters

**Concepts:**
- **Chunk** - Self-contained unit of documentation (code example, explanation, API method)
- **Metadata** - Searchable attributes (VTK classes used, source type, complexity)
- **Structured Chunking** - Respects semantic boundaries (don't split functions mid-code)

**Output:** `data/processed/*.jsonl` - ~131,000 chunks ready for indexing

📖 [Detailed docs](prepare-corpus/README.md)

---

### **2. Index Building** (`build-indexes/`)

**Purpose:** Create searchable vector + keyword index in Qdrant

**Components:**
- `build_qdrant_index.py` - Builds Qdrant collection
  - Embeds chunks using sentence-transformers
  - Creates payload with metadata for filtering
  - Batched upload for performance

**Concepts:**
- **Vector Search** - Semantic similarity using embeddings
- **Hybrid Search** - Combines vector + BM25 keyword search
- **Metadata Filtering** - Filter by source type, complexity, VTK classes

**Output:** Qdrant collection `vtk_docs` with ~131K vectors

📖 [Detailed docs](build-indexes/README.md)

---

### **3. Retrieval Pipeline** (`retrieval-pipeline/`)

**Purpose:** Find most relevant documentation for a query

**Components:**
- `vtk_retrieval_pipeline.py` - Multi-stage retrieval
  - Query rewriting (VTK-specific normalization)
  - Vector search (initial candidates)
  - Cross-encoder reranking (precision refinement)

**Concepts:**
- **Query Rewriting** - Normalize VTK class names, expand acronyms, add synonyms
- **Cross-Encoder Reranking** - Re-score candidates with transformer model
- **Top-K Retrieval** - Return most relevant K documents per query/step

**Output:** Ranked list of relevant chunks with scores

📖 [Detailed docs](retrieval-pipeline/README.md)

---

### **4. Grounding & Prompting** (`grounding-prompting/`)

**Purpose:** Build LLM prompts with retrieved context and constraints

**Components:**
- `code_generation_prompt.py` - Code generation templates
- `api_documentation_prompt.py` - API lookup templates  
- `query_decomposition_prompt.py` - Step breakdown templates
- `concept_explanation_prompt.py` - Explanation templates

**Concepts:**
- **Grounded Prompting** - Include retrieved docs as context
- **Citation Enforcement** - Require LLM to cite sources
- **Refusal Policy** - Explicitly forbid hallucinations
- **Structured Output** - Request JSON with code, explanation, citations

**Output:** Formatted prompts ready for LLM

📖 [Detailed docs](grounding-prompting/README.md)

---

### **5. LLM Generation** (`llm-generation/`)

**Purpose:** Generate code and explanations using LLMs

**Components:**
- `sequential_pipeline.py` - **Main orchestrator**
  - Multi-step query handling
  - Per-step retrieval & generation
  - Result assembly and deduplication
- `llm_client.py` - Multi-provider LLM interface
  - OpenAI, Anthropic, Google, local models
  - Streaming and token counting
- `code_validator.py` - Syntax and import validation
- `security_validator.py` - Security checks (blocks dangerous patterns)

**Concepts:**
- **Sequential Generation** - Generate code step-by-step for complex queries
- **Multi-Provider** - Abstract LLM interface for flexibility
- **Validation** - Catch errors before execution
- **Refinement** - Iterative fixing with LLM

**Output:** Validated Python code + explanation + citations

📖 [Detailed docs](llm-generation/README.md)

---

### **6. API Validation** (`api-mcp/`)

**Purpose:** Detect VTK API hallucinations (non-existent classes/methods)

**Components:**
- `vtk_validator.py` - MCP-based validator
  - Tracks VTK classes and methods
  - Validates imports, method calls
  - Suggests corrections for errors
- `vtk_api_server.py` - VTK API database
  - 2,900+ classes with methods
  - Fast fuzzy matching for suggestions

**Concepts:**
- **API Hallucination** - LLM invents methods/classes that don't exist
- **Static Analysis** - Parse code to find API usage
- **Automatic Fixing** - Replace hallucinations with correct API calls
- **MCP Protocol** - Model Context Protocol for validation

**Output:** API validation results with fix suggestions

**Results:** 
- Detects 10% of generated code has API errors
- 90%+ automatic fix rate
- <5ms validation per query

📖 [Detailed docs](api-mcp/README.md)

---

### **7. Post-Processing** (`post-processing/`)

**Purpose:** Parse LLM responses into structured components

**Components:**
- `json_response_processor.py` - JSON parser and validator
  - Extracts code, explanation, citations
  - Validates structure
  - Optional LLM enrichment

**Concepts:**
- **Structured Output** - Predictable JSON format
- **Component Extraction** - Separate code, explanation, metadata
- **Enrichment** - Optionally enhance explanations with LLM

**Output:** Structured response object

📖 [Detailed docs](post-processing/README.md)

---

### **8. Visual Validation** (`visual_testing/`, `evaluation/`)

**Purpose:** Execute generated code to verify correctness

**Components:**
- `visual_evaluator.py` - Docker sandbox executor
  - Runs code in isolated container
  - Captures visual output (PNG images)
  - Detects execution errors
- `docker_sandbox.py` - Container management
  - Timeout enforcement
  - Memory limits
  - Cleanup

**Concepts:**
- **Sandbox Execution** - Isolated environment for safety
- **Visual Regression** - Compare output images (SSIM similarity)
- **Timeout** - Kill runaway code (30s default)
- **Baseline Comparison** - Detect visual changes

**Output:** Execution results + optional visual output

**Performance:**
- 100% execution success (10/10 test queries)
- 80% visual output generation
- 3.3s average execution time

📖 [Detailed docs](visual_testing/README.md)

---

## Testing (`tests/`)

The repository includes comprehensive testing at multiple levels:

### **Unit Tests**

Test individual components in isolation using mocks:

- `tests/llm-generation/` - Pipeline orchestration, validation
- `tests/grounding-prompting/` - Prompt templates
- `tests/post-processing/` - JSON parsing
- `tests/api_mcp/` - API validation
- `tests/visual_testing/` - Docker sandbox

**Run unit tests:**
```bash
./run_all_tests.sh
```

**Results:** 20+ unit tests covering all major components

### **Integration Tests**

Test end-to-end flows with real components:

- `tests/integration/` - Complete query pipeline
- `tests/evaluation/` - Full RAG workflow
- `tests/test_query.py` - `query.py` CLI interface (20 tests)

**Key test:** `test_query.py` validates:
- Query routing (code vs API vs explanation)
- Visual testing integration
- Output file generation
- CLI argument parsing
- Error handling

### **Visual Regression Tests**

Verify generated code produces expected visual output:

- Compare generated images to baselines
- SSIM (Structural Similarity) scoring
- Detect rendering regressions

**Run with visual tests:**
```bash
RUN_VISUAL_TESTS=1 ./run_all_tests.sh
```

### **Test Philosophy**

1. **Fast Unit Tests** - Mock external dependencies (LLM, Qdrant, Docker)
2. **Integration Tests** - Use real components but small datasets
3. **Visual Tests** - Opt-in (require Docker, slower)
4. **CI-Ready** - All tests runnable in automated pipelines

---

## Evaluation (`evaluation/`)

Comprehensive metrics to measure RAG pipeline quality:

### **Purpose**

Answer the key questions:
1. **Does retrieval find relevant docs?** (Retrieval metrics)
2. **Is generated code correct?** (Code quality metrics)
3. **Does code execute?** (Execution metrics)
4. **Are there API hallucinations?** (API validation metrics)
5. **Does visual output match expectations?** (Visual regression metrics)

### **Components**

- `test_set_builder.py` - Generate test queries from corpus
- `retrieval_metrics.py` - Recall@K, nDCG@K, MRR
- `end_to_end_metrics.py` - Code exactness, correctness
- `evaluator.py` - Main evaluation orchestrator
- `visual_evaluator.py` - Execution and visual testing

### **Metrics Computed**

#### **1. Retrieval Metrics** (Information Retrieval)

Measure how well the system finds relevant documentation:

- **Recall@K** - % of relevant docs in top K results
  - Target: >80% Recall@5 (most relevant docs in top 5)
- **nDCG@K** - Normalized Discounted Cumulative Gain (ranking quality)
  - Target: >0.7 (good ranking of results)
- **MRR** - Mean Reciprocal Rank (position of first relevant doc)
  - Target: >0.8 (relevant doc near top)

**How measured:** Compare retrieved chunks to ground-truth relevant chunks from gold examples

#### **2. Code Quality Metrics**

Measure correctness of generated code:

- **Exactness** - Similarity to gold code (BLEU-like)
  - Target: >20% (code style varies, but should be similar)
- **Correctness** - Has all necessary components (imports, classes, methods)
  - Target: >75% (most code should be complete)
- **Syntax Validity** - Parses without syntax errors
  - Target: 100% (always generate valid Python)

**How measured:** Compare generated code to gold examples using AST parsing and text similarity

#### **3. Validation Metrics**

Measure automatic error detection and fixing:

- **Syntax Validation Coverage** - % of code validated
  - Target: 100% (always validate)
- **Syntax Error Rate** - % with syntax errors detected
  - Target: <5% (rare errors)
- **Validation Fix Rate** - % of errors automatically fixed
  - Target: >90% (most errors fixed by retry)

**How measured:** Track validation attempts, errors found, and successful fixes

#### **4. API Validation Metrics** (Hallucination Detection)

Measure VTK API correctness:

- **API Error Detection Rate** - % of code with hallucinations
  - Current: ~10% (LLMs occasionally hallucinate)
- **API Fix Rate** - % of detected errors automatically corrected
  - Current: 90%+ (most hallucinations fixed)
- **Validation Speed** - Time to validate per query
  - Current: <5ms (near-instant)

**How measured:** Parse generated code, check all VTK API calls against database, track fix success

**Results:** Automatic API validation catches 10% of generations with hallucinations and fixes 90%+ before execution

#### **5. Execution Metrics**

Measure if code actually runs:

- **Execution Success Rate** - % of code that runs without errors
  - **Current: 100% (10/10 test queries)** ✅
- **Execution Time** - Average time to execute
  - Current: 3.3s average
- **Timeout Rate** - % of code that exceeds 30s timeout
  - Current: 0% (after query refinements)

**How measured:** Execute code in Docker sandbox, capture exit codes and timing

#### **6. Visual Validation Metrics**

Measure visual output quality:

- **Visual Output Rate** - % of successful executions that produce images
  - Current: 80% (8/10 visualization queries, 2/10 are file operations)
- **Visual Regression Rate** - % matching baseline images
  - Current: 37.5% exact matches (SSIM >0.95)
- **Average Similarity** - Mean SSIM score vs baselines
  - Current: 0.38 (different camera angles/colors, but correct geometry)

**How measured:** Compare generated PNG images to baseline images using SSIM (Structural Similarity Index)

**Note:** Visual regression is loose - different colors/angles are OK as long as geometry is correct

### **Running Evaluation**

```bash
# Retrieval only (fast, free)
python evaluation/evaluator.py --mode retrieval --num-examples 50

# End-to-end (uses LLM, costs money)
python evaluation/evaluator.py --mode end-to-end --num-examples 10

# With visual testing (requires Docker)
CREATE_BASELINES=1 RUN_VISUAL_TESTS=1 python evaluation/evaluator.py \
  --mode end-to-end \
  --num-examples 10 \
  --enable-visual-testing
```

### **Latest Results**

**Test Set:** 10 VTK visualization queries  
**LLM:** Claude Sonnet 4  
**Date:** October 2025

| Metric | Result | Target |
|--------|--------|--------|
| **Execution Success** | **100%** ✅ | >80% |
| **Visual Output Generation** | 80% | >70% |
| **API Validation Coverage** | 100% | 100% |
| **API Hallucination Rate** | 10% | <15% |
| **API Fix Rate** | 90%+ | >85% |
| **Avg Execution Time** | 3.3s | <10s |
| **Visual Regression Pass** | 37.5% | >30% |
| **Visual Similarity (Avg)** | 0.38 | >0.3 |

**Key Achievement:** 100% execution success rate after query refinements (explicit instructions about data loading, avoiding Python loops, specifying filters)

📖 [Detailed docs](evaluation/README.md)

---

## Query Interface (`query.py`)

The unified entry point to the system:

```bash
# Code generation
python query.py "How do I create a cylinder in VTK?"

# API documentation
python query.py "What methods does vtkPolyDataMapper have?"

# With visual validation
python query.py "Create a red sphere" --visual-test

# Save to file
python query.py "Show me a cone example" --output result.json

# Enhanced explanations
python query.py "Explain the VTK pipeline" --enrich
```

**See [USAGE.md](USAGE.md) for complete documentation.**

---

## Quick Start

### 1. Setup

```bash
./setup.sh
source .venv/bin/activate
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add LLM API key
```

### 3. Add Data

Place JSONL files in `data/raw/`:
- `vtk-python-docs.jsonl` (~2,900 classes)
- `vtk-python-examples.jsonl` (~850 examples)
- `vtk-python-tests.jsonl` (~900 tests)

### 4. Build Index

```bash
# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Run pipeline (corpus + index only)
python build.py

# Or: Build with Docker image for visual testing
python build.py --build-docker
```

**Time:** 
- Without Docker: ~5-10 minutes
- With Docker: ~15-25 minutes (first build downloads VTK + dependencies)

**Options:**
- `--build-docker` - Also build Docker image for visual testing (enables `--visual-test` flag in `query.py`)

### 5. Query

```bash
python query.py "How do I create a cylinder in VTK?"
```

---

## Project Status

✅ **Production Ready**
- 100% execution success rate on test queries
- Comprehensive testing (unit, integration, visual)
- Complete documentation
- Multi-provider LLM support
- API hallucination detection & fixing

**Current Limitations:**
- Visual regression is loose (different colors/angles OK)
- LLM costs ~$0.001-0.01 per query
- Requires Qdrant + Docker for full functionality
- Index rebuild takes ~10 minutes

**Future Work:**
- Streaming responses
- Caching layer for common queries
- UI/web interface
- Fine-tuned embeddings for VTK domain
- Expanded test coverage (>10 queries)

---

## Documentation

- **[USAGE.md](USAGE.md)** - Complete query.py guide with examples
- **[prepare-corpus/README.md](prepare-corpus/README.md)** - Corpus preparation
- **[build-indexes/README.md](build-indexes/README.md)** - Index building
- **[retrieval-pipeline/README.md](retrieval-pipeline/README.md)** - Retrieval system
- **[grounding-prompting/README.md](grounding-prompting/README.md)** - Prompt engineering
- **[llm-generation/README.md](llm-generation/README.md)** - LLM orchestration
- **[api-mcp/README.md](api-mcp/README.md)** - API validation
- **[post-processing/README.md](post-processing/README.md)** - Response parsing
- **[visual_testing/README.md](visual_testing/README.md)** - Docker sandbox
- **[evaluation/README.md](evaluation/README.md)** - Metrics & testing

---

## License

MIT License - provided as-is for VTK documentation processing.
