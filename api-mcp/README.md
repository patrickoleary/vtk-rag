# VTK API Validation via MCP

Post-generation validation of VTK Python code using Model Context Protocol (MCP).

## Overview

This module provides automatic validation of generated VTK code to catch API hallucinations:

- ✅ **Direct API lookups** - No vector search overhead, exact class/method verification
- ✅ **Method existence validation** - Detects when LLM invents non-existent methods
- ✅ **Import validation** - Verifies VTK classes are imported from correct modules
- ✅ **Fast in-memory index** - Loads ~2,900 VTK classes at startup
- ✅ **Structured error reporting** - Clear error messages with suggestions
- ✅ **MCP server integration** - Can be used standalone or as MCP server

---

## Quick Start

### 1. Install Dependencies

```bash
cd api-mcp
pip install -r requirements.txt
```

### 2. Run Example

```bash
python example_usage.py
```

This will validate sample VTK code and show error detection in action.

---

## File Inventory

### Core Scripts

| File | Purpose | Line Count |
|------|---------|------------|
| `vtk_api_server.py` | MCP server with VTK API index (clean library) | ~570 lines |
| `vtk_validator.py` | Code validation logic (clean library) | ~380 lines |

### Supporting Files

| File | Purpose | Line Count |
|------|---------|------------|
| `example_usage.py` | Demonstrates validation capabilities | ~165 lines |
| `requirements.txt` | Python dependencies | - |
| `README.md` | This file | - |

### Tests

Tests are located in `../tests/api_mcp/` (22 tests total):
- `test_vtk_api_server.py` - 11 tests for MCP server functionality
- `test_validation_integration.py` - 11 tests for validation workflow

---

## Usage

### Standalone Validation

```python
from pathlib import Path
from vtk_api_server import VTKAPIIndex
from vtk_validator import VTKCodeValidator

# Load VTK API index
api_docs_path = Path("../data/raw/vtk-python-docs.jsonl")
api_index = VTKAPIIndex(api_docs_path)

# Create validator
validator = VTKCodeValidator(api_index)

# Validate code
code = """
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage

stencil = vtkImageStencilToImage()
stencil.SetOutputWholeExtent([0, 10, 0, 10, 0, 10])  # ❌ Method doesn't exist!
"""

result = validator.validate_code(code)

if not result.is_valid:
    print("Validation errors found:")
    print(result.format_errors())
    # Output:
    # 1. UNKNOWN_METHOD: Method 'SetOutputWholeExtent' not found on class 'vtkImageStencilToImage'
    #    Suggestion: Did you mean SetOutputOrigin or SetOutputSpacing?
```

### As MCP Server

Add to your MCP settings (e.g., Claude Desktop config):

```json
{
  "mcpServers": {
    "vtk-api": {
      "command": "python",
      "args": [
        "/path/to/vtk-rag/api-mcp/vtk_api_server.py",
        "--api-docs",
        "/path/to/vtk-rag/data/raw/vtk-python-docs.jsonl"
      ]
    }
  }
}
```

---

## Architecture

### VTKAPIIndex (vtk_api_server.py)

Fast in-memory index of all VTK classes and methods:

```
VTKAPIIndex
├── Classes Dict: {class_name → {module, methods, docs}}
├── Modules Dict: {module_name → [class_names]}
└── Load Time: <1 second for ~2,900 classes
```

**Key Methods:**
- `get_class_info(class_name)` - Get module and documentation
- `search_classes(query)` - Search by name or keyword
- `get_module_classes(module)` - List classes in module
- `class_exists(class_name)` - Check if class exists

### VTKCodeValidator (vtk_validator.py)

AST-based validation of generated Python code:

```
VTKCodeValidator
├── Parse Code: Uses Python's ast module
├── Extract VTK Usage:
│   ├── Import statements
│   ├── Class instantiations
│   └── Method calls
├── Validate Against Index:
│   ├── Check classes exist
│   ├── Check imports correct
│   └── Check methods exist
└── Generate Error Report
```

**Validation Types:**
1. **Import Validation** - Verifies module paths
2. **Class Validation** - Checks class existence
3. **Method Validation** - Detects hallucinated methods

---

## MCP Tools Provided

When running as MCP server, provides these tools:

### 1. `vtk_get_class_info`
Get complete information about a VTK class.

**Input:**
```json
{
  "class_name": "vtkPolyDataMapper"
}
```

**Output:**
```json
{
  "class_name": "vtkPolyDataMapper",
  "module": "vtkmodules.vtkRenderingCore",
  "content_preview": "vtkPolyDataMapper - map vtkPolyData to graphics primitives..."
}
```

### 2. `vtk_search_classes`
Search for VTK classes by name or keyword.

**Input:**
```json
{
  "query": "reader",
  "limit": 5
}
```

**Output:**
```json
[
  {
    "class_name": "vtkSTLReader",
    "module": "vtkmodules.vtkIOGeometry",
    "description": "Read ASCII or binary stereo lithography files."
  }
]
```

### 3. `vtk_validate_import`
Validate and correct VTK import statements.

**Input:**
```json
{
  "import_statement": "from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper"
}
```

**Output:**
```json
{
  "valid": false,
  "message": "Incorrect module. 'vtkPolyDataMapper' is in 'vtkmodules.vtkRenderingCore'",
  "suggested": "from vtkmodules.vtkRenderingCore import vtkPolyDataMapper"
}
```

### 4. `vtk_get_method_info`
Get documentation for a specific method.

**Input:**
```json
{
  "class_name": "vtkPolyDataMapper",
  "method_name": "SetInputData"
}
```

**Output:**
```json
{
  "class_name": "vtkPolyDataMapper",
  "method_name": "SetInputData",
  "documentation": "SetInputData(vtkDataObject) - Set the input data..."
}
```

---

## Benefits Over RAG Retrieval

| Aspect | RAG Retrieval | MCP Validation |
|--------|---------------|----------------|
| **Speed** | Vector search + reranking | Direct hash lookup (instant) |
| **Accuracy** | Semantic similarity (can drift) | Exact API match (100%) |
| **Coverage** | Top-K only (~10 results) | All ~2,900 classes available |
| **Tokens** | Consumes prompt tokens | Tool calls (minimal cost) |
| **Errors** | Silent hallucinations | Explicit error messages |

---

## Validation Examples

### Example 1: Method Hallucination (CAUGHT ✅)

**Generated Code:**
```python
stencil = vtkImageStencilToImage()
stencil.SetOutputWholeExtent([0, 10, 0, 10, 0, 10])  # ❌ Doesn't exist!
```

**Validation Error:**
```
UNKNOWN_METHOD: Method 'SetOutputWholeExtent' not found on class 'vtkImageStencilToImage'
Suggestion: Did you mean SetOutputOrigin or SetOutputSpacing?
```

### Example 2: Wrong Import Module (CAUGHT ✅)

**Generated Code:**
```python
from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper  # ❌ Wrong module!
```

**Validation Error:**
```
IMPORT_ERROR: 'vtkPolyDataMapper' is not in module 'vtkmodules.vtkCommonDataModel'
Correct import: from vtkmodules.vtkRenderingCore import vtkPolyDataMapper
```

### Example 3: Non-existent Class (CAUGHT ✅)

**Generated Code:**
```python
converter = vtkImageDataToPolyDataConverter()  # ❌ Class doesn't exist!
```

**Validation Error:**
```
UNKNOWN_CLASS: Class 'vtkImageDataToPolyDataConverter' not found in VTK
Suggestion: Did you mean vtkImageDataGeometryFilter?
```

---

## Integration with Pipeline

### Planned Integration

This validator will be integrated into `sequential_pipeline.py`:

```python
from api_mcp.vtk_validator import VTKCodeValidator

class SequentialPipeline:
    def __init__(self, enable_api_validation=True):
        self.enable_api_validation = enable_api_validation
        if enable_api_validation:
            self.validator = VTKCodeValidator(api_index)
    
    def _validate_api(self, code: str) -> dict:
        """Validate generated code for API errors"""
        result = self.validator.validate_code(code)
        return {
            'valid': result.is_valid,
            'errors': result.errors,
            'formatted_errors': result.format_errors()
        }
```

**Pipeline Flow (Future):**
```
1. Generate code with LLM
2. Fix common VTK issues (_fix_common_vtk_issues)
3. ✨ NEW: Validate with VTKCodeValidator
4. Security validation (optional)
5. LLM validation (optional)
6. Return validated code
```

---

## Testing

Tests are located in `../tests/api_mcp/` and use unittest framework.

**Test Coverage:** 22 tests across 2 files
- `test_vtk_api_server.py` - 11 tests for MCP server functionality
- `test_validation_integration.py` - 11 tests for code validation workflow

```bash
# Run all project tests (includes api_mcp)
bash run_all_tests.sh

# Run just API MCP tests
cd tests/api_mcp
python test_vtk_api_server.py
python test_validation_integration.py

# Or using pytest
cd tests
pytest api_mcp/ -v
```

---

## Performance

| Metric | Value |
|--------|-------|
| **Index Load Time** | <1 second |
| **Class Lookup** | <1 ms |
| **Method Validation** | <5 ms per method |
| **Full Code Validation** | <50 ms (typical) |
| **Memory Usage** | ~50 MB (index in memory) |

---

## Data Source

**Input:** `data/raw/vtk-python-docs.jsonl`

Each line is a VTK class documentation in JSON format:

```json
{
  "class": "vtkPolyDataMapper",
  "module": "vtkmodules.vtkRenderingCore",
  "methods": ["SetInputData", "GetInput", "Update", ...],
  "documentation": "Full class documentation..."
}
```

**Coverage:** ~2,900 VTK classes from VTK Python API

---

## Future Enhancements

- [ ] **Method signature validation** - Check parameter types and counts
- [ ] **Deprecation warnings** - Flag deprecated VTK methods
- [ ] **Pipeline validation** - Verify data flow compatibility
- [ ] **Auto-fix suggestions** - Generate corrected code automatically
- [ ] **Performance profiling** - Track validation overhead
- [ ] **Cache layer** - Cache frequent lookups for speed

---

## See Also

- **llm-generation/** - Main code generation pipeline
- **grounding-prompting/** - Prompt templates with grounding
- **evaluation/** - Evaluation metrics and test queries
- **mcp.md** - Full MCP integration plan

---

**Status:** Phases 1-2 complete. Integration (Phase 3) and evaluation (Phase 4) pending.  
**Next:** Test with `example_usage.py`, then integrate into `sequential_pipeline.py`.
