#!/usr/bin/env python3
"""
Add pythonic API explanation to corpus

This creates a synthetic explanation chunk documenting the pythonic import style
so the LLM understands the API pattern explicitly.
"""

import json
from pathlib import Path

# Create pythonic API primer
primer = {
    "title": "VTK Pythonic API Overview",
    "category": "Python API",
    "description": """
# VTK Pythonic Import Style

VTK offers two import styles for Python:

## Modern Pythonic Style (Recommended)

```python
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkRenderingCore import vtkActor
```

**Benefits:**
- Faster imports (only loads needed modules)
- Smaller memory footprint
- Clearer dependencies
- Better for large applications

**Pattern:** `from vtkmodules.<module> import <ClassName>`

**Common Modules:**
- `vtkmodules.vtkCommonCore` - Basic data structures (vtkPoints, vtkDataArray)
- `vtkmodules.vtkCommonDataModel` - Data objects (vtkPolyData, vtkImageData)
- `vtkmodules.vtkFiltersCore` - Core filters (vtkGlyph3D, vtkContourFilter)
- `vtkmodules.vtkFiltersGeneral` - General filters (vtkClipPolyData, vtkWarpVector)
- `vtkmodules.vtkFiltersSources` - Source objects (vtkSphereSource, vtkConeSource)
- `vtkmodules.vtkRenderingCore` - Rendering (vtkActor, vtkMapper, vtkRenderer)
- `vtkmodules.vtkIOLegacy` - File I/O (vtkPolyDataReader, vtkStructuredGridReader)

## Old Monolithic Style (Legacy)

```python
import vtk
points = vtk.vtkPoints()
glyph = vtk.vtkGlyph3D()
```

**When to use:** Older codebases, quick prototyping, backwards compatibility.

## Hybrid Approach

```python
from vtkmodules.vtkCommonCore import vtkPoints
import vtkmodules.vtkRenderingOpenGL2  # Enable OpenGL2 backend
```

## Common User Queries

- How do I import VTK classes in Python?
- What is the vtkmodules import style?
- Difference between import vtk and from vtkmodules?
- How to use pythonic VTK API?
- What VTK modules are available?
- Best practice for VTK Python imports?
""",
    "common_queries": [
        "How to import VTK classes in Python",
        "What is vtkmodules import style",
        "Difference between import vtk and from vtkmodules",
        "VTK pythonic API usage",
        "VTK Python import best practices",
        "What are VTK modules",
        "How to use pythonic VTK"
    ]
}

# Add to explanation chunks
output_file = Path('data/processed/explanation_chunks.jsonl')

# Create chunk in same format as other explanations
chunk = {
    "chunk_id": "PythonicAPI_explanation_0",
    "content": primer["description"],
    "content_type": "explanation",
    "source_type": "documentation",
    "metadata": {
        "title": primer["title"],
        "category": primer["category"],
        "common_queries": primer["common_queries"]
    }
}

# Append to file
print("Adding pythonic API primer to explanation chunks...")
with open(output_file, 'a') as f:
    f.write(json.dumps(chunk) + '\n')

print(f"✓ Added chunk: {chunk['chunk_id']}")
print(f"  Content: {len(chunk['content'])} chars")
print(f"  Queries: {len(chunk['metadata']['common_queries'])}")
print("\nNow rebuild Qdrant index to include this chunk.")
