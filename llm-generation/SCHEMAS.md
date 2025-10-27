# JSON Schemas for LLM Pipeline Communication

This document defines the JSON schemas used for structured communication between the application and LLM throughout the VTK RAG pipeline. All schemas are implemented as Python dataclasses in `schemas.py`.

---

## Overview

The pipeline uses structured JSON for three main phases:
1. **Decomposition** - Break complex queries into logical steps
2. **Generation** - Generate code for each step with retrieved documentation
3. **Validation** - Fix syntax/runtime errors (optional)

**Benefits:**
- ✅ Structured, parseable responses
- ✅ Schema validation
- ✅ Type safety
- ✅ Clear contract between application and LLM

---

## 1. Decomposition Phase

### DecompositionInput

**Purpose:** Send query to LLM for analysis and step-by-step planning.

**Python Class:** `DecompositionInput`

**Schema:**
```json
{
  "query": "How can I create a basic rendering of a polygonal cylinder in VTK?",
  "instructions": {
    "role": "You are an expert VTK application developer...",
    "task": "Analyze this query and break it into logical tasks",
    "think_through": [
      "What is the user actually trying to accomplish?",
      "What specific libraries or tools are mentioned?",
      "What specific data files are mentioned?",
      "Does this require visualization?",
      "What are the 3-8 concrete steps needed?"
    ],
    "requirements": [
      "DO NOT force visualization if not needed",
      "DO NOT ignore specific libraries mentioned",
      "DO NOT write code - only analyze and plan"
    ]
  }
}
```

**Fields:**
- `query` (string, required) - The user's query
- `instructions` (object, required) - Instructions for LLM decomposition
  - `role` (string) - System role/persona
  - `task` (string) - What to do
  - `think_through` (array) - Questions to consider
  - `requirements` (array) - Constraints and rules

---

### DecompositionOutput

**Purpose:** LLM returns structured plan with steps.

**Python Class:** `DecompositionOutput`

**Schema:**
```json
{
  "understanding": "User wants to create a basic rendering of a polygonal cylinder in VTK",
  "requires_visualization": true,
  "libraries_needed": ["vtk"],
  "data_files": [],
  "steps": [
    {
      "step_number": 1,
      "description": "Create a polygonal cylinder geometry source",
      "search_query": "VTK vtkCylinderSource create polygonal cylinder geometry",
      "focus": "geometry"
    },
    {
      "step_number": 2,
      "description": "Create a mapper to convert geometry to graphics",
      "search_query": "VTK vtkPolyDataMapper map geometry data",
      "focus": "rendering"
    }
  ]
}
```

**Fields:**
- `understanding` (string, required) - LLM's interpretation of the query
- `requires_visualization` (boolean, required) - Whether visualization is needed
- `libraries_needed` (array, required) - External libraries (pandas, numpy, vtk, etc.)
- `data_files` (array, required) - Data files mentioned in query
- `steps` (array, required) - List of Step objects

**Step Object:**
- `step_number` (int, required) - Sequential step number
- `description` (string, required) - Human-readable step description
- `search_query` (string, required) - Optimized query for retrieving docs
- `focus` (string, required) - Category: "geometry", "rendering", "io", "filtering", etc.

**Validation:** `validate_decomposition_output(data)`

---

## 2. Generation Phase (Per-Step)

### GenerationInput

**Purpose:** Provide LLM with context to generate code for current step.

**Python Class:** `GenerationInput`

**Schema:**
```json
{
  "original_query": "How can I create a basic rendering of a polygonal cylinder in VTK?",
  "overall_understanding": "User wants to create a basic rendering...",
  "overall_plan": {
    "total_steps": 5,
    "current_step_number": 2,
    "steps": [
      {
        "step_number": 1,
        "description": "Create a polygonal cylinder geometry source",
        "status": "completed"
      },
      {
        "step_number": 2,
        "description": "Create a mapper to convert geometry",
        "status": "current"
      }
    ]
  },
  "current_step": {
    "step_number": 2,
    "description": "Create a mapper to convert cylinder geometry",
    "focus": "rendering"
  },
  "previous_steps": [
    {
      "step_number": 1,
      "understanding": "Created a vtkCylinderSource with 8-sided resolution",
      "imports": ["from vtkmodules.vtkFiltersSources import vtkCylinderSource"],
      "code": "cylinder = vtkCylinderSource()\ncylinder.SetResolution(8)"
    }
  ],
  "documentation": [
    {
      "index": 1,
      "chunk_id": "CylinderExample_code_0",
      "content_type": "CODE",
      "content": "from vtkmodules.vtkRenderingCore import vtkPolyDataMapper..."
    },
    {
      "index": 2,
      "chunk_id": "vtkPolyDataMapper_api_0",
      "content_type": "API_DOC",
      "content": "vtkPolyDataMapper - map vtkPolyData to graphics primitives..."
    }
  ],
  "instructions": {
    "task": "Generate Python code for the current step using pythonic VTK API",
    "citation_policy": "MANDATORY",
    "code_style": "pythonic",
    "requirements": [
      "Use pythonic VTK API (property setters, constructor args)",
      "Import only what you need from vtkmodules.* submodules",
      "Build upon previous step code - reference existing variables",
      "Cite documentation chunks using [N] notation",
      "Include clear comments explaining VTK pipeline connections"
    ]
  }
}
```

**Fields:**
- `original_query` (string, required) - Original user query
- `overall_understanding` (string, required) - From decomposition output
- `overall_plan` (object, required) - OverallPlan with progress
- `current_step` (object, required) - CurrentStepInfo being generated
- `previous_steps` (array, required) - Results from completed steps
- `documentation` (array, required) - Retrieved DocumentationChunk objects
- `instructions` (object, required) - Generation instructions

**Supporting Objects:**

**OverallPlan:**
- `total_steps` (int)
- `current_step_number` (int)
- `steps` (array) - Steps with status markers

**CurrentStepInfo:**
- `step_number` (int)
- `description` (string)
- `focus` (string)

**PreviousStepResult:**
- `step_number` (int)
- `understanding` (string)
- `imports` (array)
- `code` (string)

**DocumentationChunk:**
- `index` (int) - Citation number [1], [2], etc.
- `chunk_id` (string)
- `content_type` (string) - "CODE", "API_DOC", "CONCEPT", etc.
- `content` (string)

---

### GenerationOutput

**Purpose:** LLM returns generated code for the step.

**Python Class:** `GenerationOutput`

**Schema:**
```json
{
  "step_number": 2,
  "understanding": "Creates a vtkPolyDataMapper to convert the cylinder geometry into renderable graphics primitives",
  "imports": [
    "from vtkmodules.vtkRenderingCore import vtkPolyDataMapper"
  ],
  "code": "mapper = vtkPolyDataMapper()\nmapper.SetInputConnection(cylinder.GetOutputPort())",
  "citations": [1, 3]
}
```

**Fields:**
- `step_number` (int, required) - Step number being generated
- `understanding` (string, required) - What this step accomplishes
- `imports` (array, required) - Import statements needed for this step
- `code` (string, required) - Python code for this step
- `citations` (array, required) - Citation indices [1], [2], etc.

**Validation:** `validate_generation_output(data)`

---

## 3. Validation Phase (Optional)

### ValidationInput

**Purpose:** Send code with errors to LLM for correction.

**Python Class:** `ValidationInput`

**Schema:**
```json
{
  "task": "Fix syntax and runtime errors in the generated VTK code",
  "original_query": "How can I create a basic rendering of a polygonal cylinder in VTK?",
  "generated_code": "from vtkmodules.vtkFiltersSources import vtkCylinderSource\nfrom vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor\n\ncylinder = vtkCylinderSource()\n...\nactor.SetMapper(mapper)",
  "validation_errors": [
    {
      "type": "AttributeError",
      "message": "'vtkActor' object has no attribute 'SetMapper'",
      "line": 10,
      "context": "actor.SetMapper(mapper)"
    },
    {
      "type": "NameError",
      "message": "name 'vtkRenderer' is not defined",
      "line": 12,
      "context": "renderer.AddActor(actor)"
    }
  ],
  "instructions": {
    "task": "Fix the errors in the code",
    "requirements": [
      "Fix ONLY the specific errors listed",
      "Do NOT rewrite the entire code",
      "Do NOT change working code",
      "Add missing imports if needed",
      "Fix incorrect method calls",
      "Maintain the original code structure and style"
    ],
    "output_format": "Return JSON with: fixed_code, changes_made"
  }
}
```

**Fields:**
- `task` (string, required) - What to do
- `original_query` (string, required) - User's original query
- `generated_code` (string, required) - Code with errors
- `validation_errors` (array, required) - List of ValidationError objects
- `instructions` (object, required) - Fixing instructions

**ValidationError:**
- `type` (string) - Error type (NameError, AttributeError, SyntaxError, etc.)
- `message` (string) - Error message
- `line` (int) - Line number where error occurs
- `context` (string) - Code snippet around error

---

### ValidationOutput

**Purpose:** LLM returns corrected code with explanations.

**Python Class:** `ValidationOutput`

**Schema:**
```json
{
  "fixed_code": "from vtkmodules.vtkFiltersSources import vtkCylinderSource\nfrom vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer\n\ncylinder = vtkCylinderSource()\ncylinder.SetResolution(8)\n\nmapper = vtkPolyDataMapper()\nmapper.SetInputConnection(cylinder.GetOutputPort())\n\nactor = vtkActor(mapper=mapper)\n\nrenderer = vtkRenderer()\nrenderer.AddActor(actor)",
  "changes_made": [
    {
      "error_type": "NameError",
      "fix": "Added 'vtkRenderer' to imports from vtkmodules.vtkRenderingCore",
      "line": 2
    },
    {
      "error_type": "AttributeError",
      "fix": "Changed 'actor.SetMapper(mapper)' to 'actor = vtkActor(mapper=mapper)' using pythonic API",
      "line": 9
    }
  ]
}
```

**Fields:**
- `fixed_code` (string, required) - Complete corrected code
- `changes_made` (array, required) - List of ValidationFix objects

**ValidationFix:**
- `error_type` (string) - Type of error fixed
- `fix` (string) - Description of the fix applied
- `line` (int) - Line where fix was applied

**Validation:** `validate_validation_output(data)`

---

## 4. New Query Type Schemas (Phase 1-3)

These schemas support the unified query system with multiple query types.

### APILookupOutput

**Purpose:** Return API documentation and usage information.

**Python Class:** `APILookupOutput`

**Schema:**
```json
{
  "response_type": "answer",
  "content_type": "api",
  "explanation": "SetMapper() assigns a vtkMapper to a vtkActor. The mapper converts geometric data into graphics primitives that can be rendered.",
  "usage_example": "actor = vtkActor()\nmapper = vtkPolyDataMapper()\nactor.SetMapper(mapper)",
  "parameters": [
    {
      "name": "mapper",
      "type": "vtkMapper",
      "description": "The mapper object to assign to this actor"
    }
  ],
  "return_value": "None",
  "related_methods": ["GetMapper", "SetProperty", "GetProperty"],
  "citations": [
    {"number": 1, "reason": "vtkActor API documentation"},
    {"number": 2, "reason": "vtkMapper usage example"}
  ],
  "confidence": "high"
}
```

**Fields:**
- `response_type` (string, required) - "answer"
- `content_type` (string, required) - "api"
- `explanation` (string, required) - Detailed API explanation
- `usage_example` (string) - Code example showing usage
- `parameters` (array) - Parameter descriptions
- `return_value` (string) - What the method returns
- `related_methods` (array) - Related methods/classes
- `citations` (array, required) - Source citations
- `confidence` (string, required) - "high", "medium", or "low"

**Validation:** `validate_api_lookup_output(data)`

---

### ExplanationOutput

**Purpose:** Return concept explanations and educational content.

**Python Class:** `ExplanationOutput`

**Schema:**
```json
{
  "response_type": "answer",
  "content_type": "explanation",
  "explanation": "The VTK pipeline is a dataflow architecture where data flows from sources through filters to mappers. Each component processes data and passes it to the next stage...",
  "key_concepts": [
    {
      "concept": "Source",
      "description": "Generates or reads data (e.g., vtkCylinderSource, vtkSTLReader)"
    },
    {
      "concept": "Filter",
      "description": "Processes and transforms data (e.g., vtkSmoothPolyDataFilter)"
    },
    {
      "concept": "Mapper",
      "description": "Converts data to graphics primitives (e.g., vtkPolyDataMapper)"
    }
  ],
  "examples": [
    "Simple pipeline: Source → Filter → Mapper → Actor → Renderer",
    "Data flow: vtkCylinderSource → vtkSmoothPolyDataFilter → vtkPolyDataMapper"
  ],
  "related_concepts": ["Dataflow architecture", "Lazy evaluation", "Pipeline execution"],
  "citations": [
    {"number": 1, "reason": "VTK User's Guide - Pipeline Architecture"},
    {"number": 2, "reason": "VTK Examples - Pipeline Patterns"}
  ],
  "confidence": "high"
}
```

**Fields:**
- `response_type` (string, required) - "answer"
- `content_type` (string, required) - "explanation"
- `explanation` (string, required) - Main explanation text
- `key_concepts` (array) - Structured concept definitions
- `examples` (array) - Example descriptions
- `related_concepts` (array) - Related topic names
- `citations` (array, required) - Source citations
- `confidence` (string, required) - "high", "medium", or "low"

**Validation:** `validate_explanation_output(data)`

---

### DataToCodeOutput

**Purpose:** Return exploratory data analysis with multiple technique suggestions.

**Python Class:** `DataToCodeOutput`

**Schema:**
```json
{
  "response_type": "answer",
  "content_type": "code",
  "data_analysis": "CSV file with x, y, z columns representing 3D point coordinates. Suitable for point cloud visualization, scatter plots, or surface reconstruction.",
  "suggested_techniques": [
    "Point cloud visualization with vtkPolyData",
    "3D scatter plot with vtkGlyph3D",
    "Surface reconstruction with vtkDelaunay3D"
  ],
  "code": "import pandas as pd\nfrom vtkmodules.vtkCommonCore import vtkPoints\nfrom vtkmodules.vtkCommonDataModel import vtkPolyData\n\ndf = pd.read_csv('points.csv')\npoints = vtkPoints()\nfor _, row in df.iterrows():\n    points.InsertNextPoint(row['x'], row['y'], row['z'])\n\npolydata = vtkPolyData()\npolydata.SetPoints(points)",
  "explanation": "This code reads the CSV file with pandas and creates a VTK point cloud [1]. The points are loaded into vtkPoints and wrapped in vtkPolyData for visualization [2].",
  "alternative_approaches": [
    {
      "technique": "3D Scatter Plot",
      "description": "Use vtkGlyph3D to show spheres at each point, better for seeing individual data points",
      "vtk_classes": ["vtkGlyph3D", "vtkSphereSource"],
      "complexity": "moderate"
    },
    {
      "technique": "Surface Reconstruction",
      "description": "Use vtkDelaunay3D to create a surface mesh from the point cloud",
      "vtk_classes": ["vtkDelaunay3D"],
      "complexity": "advanced"
    }
  ],
  "vtk_classes_used": ["vtkPoints", "vtkPolyData"],
  "data_files_used": ["points.csv"],
  "citations": [
    {"number": 1, "reason": "CSV to VTK points conversion pattern"},
    {"number": 2, "reason": "Point cloud visualization example"}
  ],
  "confidence": "high"
}
```

**Fields:**
- `response_type` (string, required) - "answer"
- `content_type` (string, required) - "code"
- `data_analysis` (string) - Analysis of data type and suitability
- `suggested_techniques` (array) - List of technique names
- `code` (string, required) - Working code for most common technique
- `explanation` (string, required) - How the code works
- `alternative_approaches` (array) - Detailed alternatives with complexity
- `vtk_classes_used` (array) - VTK classes in the code
- `data_files_used` (array) - Data files mentioned
- `citations` (array, required) - Source citations
- `confidence` (string, required) - "high", "medium", or "low"

**Alternative Approach Object:**
- `technique` (string) - Technique name
- `description` (string) - What it does
- `vtk_classes` (array) - Classes needed
- `complexity` (string) - "simple", "moderate", or "advanced"

**Validation:** `validate_data_to_code_output(data)`

---

### CodeToDataOutput

**Purpose:** Return example data files that work with provided code.

**Python Class:** `CodeToDataOutput`

**Schema:**
```json
{
  "response_type": "answer",
  "content_type": "data",
  "explanation": "This code reads STL mesh files using vtkSTLReader [1]. Several example STL files are available in the VTK testing data.",
  "code_requirements": "Code expects STL file with triangular mesh data. Any valid STL file (ASCII or binary format) will work.",
  "data_files": [
    {
      "filename": "42400-IDGH.stl",
      "description": "STL mesh file - tooth model with high detail",
      "source_example": "STL Reader Example",
      "download_url": "https://vtk.org/files/ExternalData/Testing/Data/42400-IDGH.stl",
      "file_type": "STL",
      "file_size": "~2MB"
    },
    {
      "filename": "sphere.stl",
      "description": "Simple sphere mesh for testing",
      "source_example": "Basic STL Example",
      "download_url": "https://vtk.org/files/ExternalData/Testing/Data/sphere.stl",
      "file_type": "STL",
      "file_size": "~100KB"
    }
  ],
  "vtk_classes_used": ["vtkSTLReader", "vtkPolyDataMapper"],
  "citations": [
    {"number": 1, "reason": "STL reader example with data files"}
  ],
  "confidence": "high"
}
```

**Fields:**
- `response_type` (string, required) - "answer"
- `content_type` (string, required) - "data"
- `explanation` (string, required) - What data the code needs
- `code_requirements` (string) - Description of expected data format
- `data_files` (array) - List of available data files with download info
- `vtk_classes_used` (array) - VTK classes detected in code
- `citations` (array, required) - Source citations
- `confidence` (string, required) - "high", "medium", or "low"

**Data File Object:**
- `filename` (string) - File name
- `description` (string) - What the file contains
- `source_example` (string) - Example it came from
- `download_url` (string) - URL to download
- `file_type` (string) - File extension (STL, CSV, etc.)
- `file_size` (string) - Approximate size

**Validation:** `validate_code_to_data_output(data)`

---

## 5. Final Result Schemas

These schemas aggregate all step results into a complete response.

### FinalResult

**Purpose:** Complete pipeline result with all steps assembled.

**Python Class:** `FinalResult`

**Schema:**
```json
{
  "query": "How can I create a basic rendering of a polygonal cylinder in VTK?",
  "understanding": "User wants to create a basic rendering of a polygonal cylinder in VTK",
  "requires_visualization": true,
  "libraries_needed": ["vtk"],
  "data_files": [],
  "steps": [
    {
      "step_number": 1,
      "description": "Create a polygonal cylinder geometry source",
      "understanding": "Created a vtkCylinderSource with 8-sided resolution",
      "imports": ["from vtkmodules.vtkFiltersSources import vtkCylinderSource"],
      "code": "cylinder = vtkCylinderSource()...",
      "citations": [1],
      "chunks_used": ["CylinderExample_code_0"]
    },
    {
      "step_number": 2,
      "description": "Create a mapper",
      "understanding": "Creates a vtkPolyDataMapper...",
      "imports": ["from vtkmodules.vtkRenderingCore import vtkPolyDataMapper"],
      "code": "mapper = vtkPolyDataMapper()...",
      "citations": [2, 3],
      "chunks_used": ["MapperExample_code_0", "vtkPolyDataMapper_api_0"]
    }
  ],
  "final_code": {
    "imports": "from vtkmodules.vtkFiltersSources import vtkCylinderSource\nfrom vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor...",
    "body": "cylinder = vtkCylinderSource()\ncylinder.SetResolution(8)\n\nmapper = vtkPolyDataMapper()...",
    "complete": "# Complete assembled code with imports and body"
  },
  "explanation": {
    "overview": "User wants to create a basic rendering of a polygonal cylinder in VTK",
    "imports": ["from vtkmodules.vtkFiltersSources import vtkCylinderSource", "..."],
    "steps": [
      {
        "step_number": 1,
        "description": "Create a polygonal cylinder geometry source",
        "understanding": "Created a vtkCylinderSource with 8-sided resolution",
        "code": "cylinder = vtkCylinderSource()..."
      }
    ],
    "formatted": "Complete formatted explanation with all steps"
  }
}
```

**Fields:**
- `query` (string, required) - Original query
- `understanding` (string, required) - Overall understanding
- `requires_visualization` (boolean, required)
- `libraries_needed` (array, required)
- `data_files` (array, required)
- `steps` (array, required) - List of StepResult objects
- `final_code` (object, required) - FinalCode object
- `explanation` (object, required) - Explanation object

**StepResult:**
- `step_number` (int)
- `description` (string)
- `understanding` (string)
- `imports` (array)
- `code` (string)
- `citations` (array)
- `chunks_used` (array)

**FinalCode:**
- `imports` (string) - Deduplicated imports
- `body` (string) - Assembled code body
- `complete` (string) - imports + body

**Explanation:**
- `overview` (string)
- `imports` (array)
- `steps` (array) - Step-by-step explanation
- `formatted` (string) - Human-readable formatted text

---

## Usage in Code

### Creating Schema Objects

```python
from schemas import DecompositionInput, DecompositionOutput, GenerationInput, GenerationOutput

# Create decomposition input
decomp_input = DecompositionInput(
    query="How to create a cylinder?",
    instructions={
        "role": "VTK expert",
        "task": "Analyze and break into steps"
    }
)

# Serialize to JSON for LLM
json_str = decomp_input.to_json()

# Parse LLM response
decomp_output = DecompositionOutput.from_dict(llm_response_dict)

# Access fields
print(decomp_output.understanding)
print(decomp_output.requires_visualization)
for step in decomp_output.get_steps():
    print(f"Step {step.step_number}: {step.description}")
```

### Validating LLM Responses

```python
from schemas import validate_decomposition_output, validate_generation_output

# Validate decomposition response
if validate_decomposition_output(llm_response):
    decomp = DecompositionOutput.from_dict(llm_response)
else:
    print("Invalid decomposition output")

# Validate generation response
if validate_generation_output(llm_response):
    gen = GenerationOutput.from_dict(llm_response)
else:
    print("Invalid generation output")
```

---

## Schema Validation Functions

Located in `schemas.py`:

### Original Pipeline Schemas:

**`validate_decomposition_output(data: Dict) -> bool`**
- Checks for required fields: understanding, requires_visualization, libraries_needed, data_files, steps

**`validate_generation_output(data: Dict) -> bool`**
- Checks for required fields: step_number, understanding, imports, code, citations

**`validate_validation_output(data: Dict) -> bool`**
- Checks for required fields: fixed_code, changes_made

### New Query Type Schemas (Phase 1-3):

**`validate_api_lookup_output(data: Dict) -> bool`**
- Checks for required fields: response_type, content_type, explanation, confidence, citations

**`validate_explanation_output(data: Dict) -> bool`**
- Checks for required fields: response_type, content_type, explanation, confidence, citations

**`validate_data_to_code_output(data: Dict) -> bool`**
- Checks for required fields: response_type, content_type, code, explanation, confidence, citations

**`validate_code_to_data_output(data: Dict) -> bool`**
- Checks for required fields: response_type, content_type, explanation, confidence, citations

---

## Best Practices

### 1. Always Validate
```python
if not validate_decomposition_output(response):
    raise ValueError("LLM returned invalid schema")
```

### 2. Use Type Hints
```python
def process_decomposition(output: DecompositionOutput) -> List[Step]:
    return output.get_steps()
```

### 3. Handle Missing Fields
```python
try:
    decomp = DecompositionOutput.from_dict(data)
except TypeError as e:
    logger.error(f"Missing required field: {e}")
```

### 4. Serialize for Storage
```python
# Save to file
with open('result.json', 'w') as f:
    f.write(final_result.to_json())

# Load from file
with open('result.json', 'r') as f:
    data = json.load(f)
    result = FinalResult.from_dict(data)
```

---

## See Also

- `schemas.py` - Python implementation of all schemas (now includes 4 new query type schemas)
- `llm_client.py` - LLM client with JSON generation and validation (now includes schema definitions for all 7 schemas)
- `sequential_pipeline.py` - Unified query system using these schemas (now handles 5 query types)
- `prompt_templates.py` - Prompt templates for all query types (10 methods total)
- `USAGE_GUIDE.md` - Comprehensive usage guide with examples for all query types
- Top-level `data_exchange.md` - Detailed examples with full JSON structures

## Summary of Changes (Phases 1-3)

**Phase 1:** Added 8 new prompt methods to `prompt_templates.py`
**Phase 2:** Extended `sequential_pipeline.py` with query classification and 4 new handlers
**Phase 3:** Added 4 new JSON schemas and validation functions

**New Schemas:**
- `APILookupOutput` - API documentation responses
- `ExplanationOutput` - Concept explanations
- `DataToCodeOutput` - Exploratory data queries with alternatives
- `CodeToDataOutput` - Data file finder

**All schemas now include:**
- Consistent `response_type`, `content_type`, `confidence`, `citations` fields
- Proper validation functions
- Full documentation with examples

---

## 6. Code Refinement Schemas (NEW)

These schemas support code refinement - modifying existing code instead of regenerating from scratch.

### ModificationDecompositionOutput

**Purpose:** Break down modification request into sequential steps.

**Python Class:** `ModificationDecompositionOutput`

**Schema:**
```json
{
  "understanding": "User wants to increase resolution and change color to blue",
  "modification_steps": [
    {
      "step_number": 1,
      "description": "Increase cylinder resolution to 50",
      "requires_retrieval": false
    },
    {
      "step_number": 2,
      "description": "Change actor color to blue",
      "requires_retrieval": false
    }
  ],
  "preserved_elements": ["variable names", "code structure", "existing configuration"]
}
```

**Fields:**
- `understanding` (string, required) - What the user wants to change
- `modification_steps` (array, required) - Sequential modification steps
  - `step_number` (int) - Step number
  - `description` (string) - What to modify
  - `requires_retrieval` (boolean) - Whether documentation retrieval needed
- `preserved_elements` (array, required) - What should NOT be changed

**Validation:** `validate_modification_decomposition_output(data)`

---

### CodeModificationOutput

**Purpose:** Return modified code with detailed change tracking.

**Python Class:** `CodeModificationOutput`

**Schema:**
```json
{
  "modifications": [
    {
      "step_number": 1,
      "modification": "Increased resolution",
      "explanation": "Changed SetResolution from 8 to 50 for smoother appearance",
      "code_changed": "cylinder.SetResolution(50)",
      "code_added": "",
      "variable_affected": "cylinder"
    },
    {
      "step_number": 2,
      "modification": "Changed color to blue",
      "explanation": "Set actor color property to blue (0, 0, 1)",
      "code_changed": "",
      "code_added": "actor.GetProperty().SetColor(0, 0, 1)",
      "variable_affected": "actor"
    }
  ],
  "updated_code": "# Complete modified code here",
  "new_imports": ["from vtkmodules.vtkCommonColor import vtkNamedColors"],
  "preserved_structure": true,
  "diff_summary": "Increased resolution to 50 and added blue color"
}
```

**Fields:**
- `modifications` (array, required) - List of modifications made
  - `step_number` (int) - Modification step
  - `modification` (string) - Short description of change
  - `explanation` (string) - Detailed why/how
  - `code_changed` (string) - Line that was changed (if applicable)
  - `code_added` (string) - Code that was added (if applicable)
  - `variable_affected` (string) - Variable name modified
- `updated_code` (string, required) - Complete modified code
- `new_imports` (array) - New import statements added
- `preserved_structure` (boolean) - Whether original structure preserved
- `diff_summary` (string) - Human-readable summary

**Validation:** `validate_code_modification_output(data)`

---

### CodeRefinementResult

**Purpose:** Complete refinement result with original code, modifications, and diff.

**Python Class:** `CodeRefinementResult`

**Schema:**
```json
{
  "response_type": "answer",
  "content_type": "code_refinement",
  "query": "Increase resolution to 50 and make it blue",
  "original_code": "# Original code before modifications",
  "code": "# Modified code (for consistency with generation)",
  "explanation": "Modification Request: Increase resolution to 50 and make it blue\n\n1. Increased resolution from 8 to 50...",
  "modifications": [
    {
      "step_number": 1,
      "modification": "Increased resolution",
      "explanation": "...",
      "code_changed": "cylinder.SetResolution(50)",
      "code_added": "",
      "variable_affected": "cylinder"
    }
  ],
  "new_imports": [],
  "citations": [
    {"number": 1, "chunk_id": "example_id", "reason": "Resolution modification pattern"}
  ],
  "chunk_ids_used": ["example_id"],
  "confidence": "high",
  "diff": "--- original\n+++ modified\n@@ -1,1 +1,1 @@\n-cylinder.SetResolution(8)\n+cylinder.SetResolution(50)"
}
```

**Fields:**
- `response_type` (string, required) - "answer"
- `content_type` (string, required) - "code_refinement"
- `query` (string, required) - Modification request
- `original_code` (string, required) - Code before modifications
- `code` (string, required) - Modified code
- `explanation` (string, required) - What changed and why
- `modifications` (array, required) - Detailed modification list
- `new_imports` (array, required) - New imports added
- `citations` (array, required) - Documentation citations
- `chunk_ids_used` (array, required) - Retrieved chunk IDs
- `confidence` (string, required) - "high", "medium", or "low"
- `diff` (string) - Unified diff showing changes

**No validator** - Used as return value wrapper

---

## 7. Enrichment Schema

### ExplanationEnrichmentOutput

**Purpose:** Enrich or generate explanations for code using LLM.

**Python Class:** `ExplanationEnrichmentOutput`

**Schema:**
```json
{
  "improved_explanation": "This code creates a cylinder visualization in VTK. The vtkCylinderSource generates the geometry with 8-sided resolution...",
  "key_points": [
    "Creates 8-sided cylinder geometry",
    "Maps geometry to graphics primitives",
    "Sets up basic rendering pipeline"
  ],
  "vtk_classes_explained": [
    {
      "name": "vtkCylinderSource",
      "purpose": "Generates polygonal cylinder geometry with configurable resolution"
    },
    {
      "name": "vtkPolyDataMapper",
      "purpose": "Converts geometric data to renderable graphics primitives"
    }
  ],
  "citations": [
    {"number": 1, "reason": "Cylinder source documentation"}
  ],
  "confidence": "high"
}
```

**Fields:**
- `improved_explanation` (string, required) - Enhanced or generated explanation
- `key_points` (array) - Main takeaways
- `vtk_classes_explained` (array) - VTK class descriptions
  - `name` (string) - Class name
  - `purpose` (string) - What it does in the code
- `citations` (array, required) - Source citations
- `confidence` (string, required) - "high", "medium", or "low"

**Validation:** `validate_explanation_enrichment_output(data)`

---

## Schema Summary

### All Schemas (11 Total)

**Original Pipeline (3):**
1. `DecompositionOutput` - Query decomposition
2. `GenerationOutput` - Code generation per step
3. `ValidationOutput` - Error correction

**Query Type Responses (4):**
4. `APILookupOutput` - API documentation
5. `ExplanationOutput` - Concept explanations
6. `DataToCodeOutput` - Data→Code with alternatives
7. `CodeToDataOutput` - Code→Data file finder

**Code Refinement (3):**
8. `ModificationDecompositionOutput` - Refinement planning
9. `CodeModificationOutput` - Modification execution
10. `CodeRefinementResult` - Complete refinement result

**Enrichment (1):**
11. `ExplanationEnrichmentOutput` - LLM-enhanced explanations

---

## Updated Validation Functions

All schemas have corresponding validators in `schemas.py`:

```python
# Original
validate_decomposition_output(data)
validate_generation_output(data)
validate_validation_output(data)

# Query types
validate_api_lookup_output(data)
validate_explanation_output(data)
validate_data_to_code_output(data)
validate_code_to_data_output(data)

# Refinement
validate_modification_decomposition_output(data)
validate_code_modification_output(data)

# Enrichment
validate_explanation_enrichment_output(data)
```

---

## Recent Changes

**October 24, 2025:**
- ✅ Added 3 code refinement schemas
- ✅ Added 1 enrichment schema
- ✅ Updated validation functions
- ✅ All 11 schemas documented with examples
