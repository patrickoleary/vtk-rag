# VTK RAG Grounding & Prompting

Centralized prompt templates with built-in grounding for the VTK RAG pipeline.

## Overview

This module provides structured JSON prompts with strict anti-hallucination measures:

- ✅ **Query decomposition** - Break complex queries into logical steps
- ✅ **Per-step generation** - Generate code with retrieved documentation context
- ✅ **Built-in grounding** - Anti-hallucination requirements in all prompts (lines 92-96)
- ✅ **Citation enforcement** - Requires `[N]` notation to documentation chunks
- ✅ **JSON-based** - Clean structured input/output for LLM communication
- ✅ **Centralized** - Single source of truth for all prompt modifications

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r grounding-prompting/requirements.txt
```

### 2. Test Prompts

```bash
cd grounding-prompting
python example_usage.py
```

### 3. Run Tests

```bash
cd tests/grounding-prompting
python test_prompt_templates.py -v
```

---

## File Inventory

### Core Scripts (Used in Pipeline)

| File | Purpose | Used By |
|------|---------|---------|
| `prompt_templates.py` | JSON prompt instructions with grounding (2 methods) | `sequential_pipeline.py` |

**Note:** All prompts include grounding requirements at **lines 92-96**.

**Prompt Methods:**

- **`get_decomposition_instructions()`** - Query → Steps
  - Breaks complex queries into logical steps
  - Returns structured instructions dict
  - Used by: `sequential_pipeline.py.decompose_query()`

- **`get_generation_instructions()`** - Step + Docs → Code
  - Generates code for each step with documentation
  - **Contains grounding at lines 92-96** 🔒
  - Used by: `sequential_pipeline.py.generate()`

### Utility Scripts

| File | Purpose | Notes |
|------|---------|-------|
| `example_usage.py` | Demonstrate prompt usage | Shows integration with pipeline |

### Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | Dependencies (none currently) |
| `README.md` | This documentation |

---

## Usage

### Get Decomposition Instructions

```python
from prompt_templates import VTKPromptTemplate

template = VTKPromptTemplate()

# Get decomposition instructions
instructions = template.get_decomposition_instructions()

# Use in pipeline
decomposition_input = {
    "query": "How can I create a basic rendering of a cylinder in VTK?",
    "instructions": instructions
}

# Send to LLM
result = llm_client.generate_json(
    prompt_data=decomposition_input,
    schema_name="DecompositionOutput"
)
```

### Get Generation Instructions (with Grounding)

```python
from prompt_templates import VTKPromptTemplate

template = VTKPromptTemplate()

# Get generation instructions (INCLUDES GROUNDING)
instructions = template.get_generation_instructions()

# Use in pipeline
generation_input = {
    "original_query": query,
    "overall_understanding": understanding,
    "overall_plan": plan_dict,
    "current_step": current_step_dict,
    "previous_steps": previous_results,
    "documentation": retrieved_chunks,
    "instructions": instructions  # Grounding at lines 92-96
}

# Send to LLM
step_response = llm_client.generate_json(
    prompt_data=generation_input,
    schema_name="GenerationOutput"
)
```

### Integration with Sequential Pipeline

```python
# In llm-generation/sequential_pipeline.py

from prompt_templates import VTKPromptTemplate

class SequentialPipeline:
    def decompose_query(self, query):
        """Uses get_decomposition_instructions()"""
        template = VTKPromptTemplate()
        
        decomposition_input = {
            "query": query,
            "instructions": template.get_decomposition_instructions()
        }
        
        return self.llm_client.generate_json(decomposition_input, "DecompositionOutput")
    
    def generate(self, query, top_k_per_step=5):
        """Uses get_generation_instructions() - includes grounding"""
        template = VTKPromptTemplate()
        
        # For each step
        generation_input = {
            "original_query": query,
            "overall_understanding": understanding,
            "overall_plan": plan,
            "current_step": step,
            "previous_steps": previous,
            "documentation": chunks,
            "instructions": template.get_generation_instructions()  # GROUNDING HERE
        }
        
        return self.llm_client.generate_json(generation_input, "GenerationOutput")
```

---

## Grounding Instructions

**Location:** `prompt_templates.py` **lines 92-96**

The generation instructions include **5 built-in grounding requirements**:

```python
# GROUNDING: Anti-hallucination and citation requirements
"Review the documentation for relevant examples",
"DO NOT hallucinate - only use patterns from documentation if they apply",
"Stay faithful to the original problem requirements",
"If the step mentions specific libraries/files, use them",
"Cite documentation using the index number [N]"
```

**What Each Rule Does:**

1. **Review documentation** - Forces LLM to look at provided chunks
2. **No hallucination** - Only use patterns found in documentation
3. **Stay faithful** - Don't deviate from original query intent
4. **Use specifics** - Respect libraries/files mentioned in query
5. **Cite sources** - Must reference documentation with [N] notation

**Why This Matters:**
- Prevents LLM from inventing VTK APIs that don't exist
- Ensures generated code is based on actual VTK documentation
- Makes responses traceable to source documentation
- Enforces grounded, factual code generation

---

## Critical VTK-Specific Warnings

**Location:** `prompt_templates.py` **lines 144-146**

In addition to general grounding, the prompts include **CRITICAL warnings** about VTK-specific issues:

### OpenGL2 Import Requirement

```python
"CRITICAL: For ANY code that uses vtkRenderWindow, vtkRenderer, or any rendering classes, you MUST start with: import vtkmodules.vtkRenderingOpenGL2",
"This OpenGL2 import MUST be the very first import before any other VTK imports",
"Without vtkRenderingOpenGL2, rendering code will crash with segmentation faults",
```

**Why this matters:**
- VTK rendering requires OpenGL2 backend to be loaded first
- Missing this import causes **segmentation faults** (crash with exit code 139)
- Must be the **first import** in the file
- **Backup:** If LLM forgets, `sequential_pipeline.py` auto-fixes it (defense in depth)

### Defense in Depth Strategy

This warning follows a **two-layer protection** approach:

1. **Prompt Layer (Preventive):** 
   - Explicit CRITICAL warning teaches LLM correct pattern
   - Reduces frequency of errors
   - Improves LLM's understanding over time

2. **Validation Layer (Corrective):**
   - `sequential_pipeline.py._fix_common_vtk_issues()` catches error
   - Auto-fixes if LLM ignores warning
   - Ensures code works even if LLM makes mistakes

**Result:** 100% success rate for rendering code

### Note on VTK Python API

**VTK Python fully supports pythonic patterns!** The following work correctly:
- ✅ `reader = vtkSTLReader(file_name='file.stl')`
- ✅ `actor = vtkActor(mapper=mapper)`
- ✅ `actor.property.edge_visibility = True`
- ✅ `interactor.render_window = window`

No warnings or fixes needed for these patterns.

---

## Modifying Grounding

To change grounding behavior:

**1. Edit the file:**
```bash
nano grounding-prompting/prompt_templates.py
```

**2. Modify lines 92-96:**
```python
# In get_generation_instructions() method:

"requirements": [
    # Code structure requirements
    "Generate ONLY the Python code statements needed for this step",
    "Put ALL import statements in the 'imports' array field",
    "Put ONLY executable code (no imports) in the 'code' field",
    "Continue from the existing code in previous_steps",
    
    # GROUNDING: Anti-hallucination and citation requirements
    "Review the documentation for relevant examples",
    "DO NOT hallucinate - only use patterns from documentation if they apply",
    "Stay faithful to the original problem requirements",
    "If the step mentions specific libraries/files, use them",
    "Cite documentation using the index number [N]",
    "YOUR NEW GROUNDING RULE HERE"  # Add your custom rule
],
```

**3. Test changes:**
```bash
cd tests/grounding-prompting
python test_prompt_templates.py -v
```

**4. Verify with evaluation:**
```bash
cd evaluation
python evaluator.py --test-set ../data/processed/test_set_augmented.jsonl --mode end-to-end --num-examples 5
```

---

## Testing

Unit tests verify prompt structure and grounding:

```bash
cd tests/grounding-prompting
python test_prompt_templates.py -v
```

**Tests verify:**
- ✅ Decomposition instructions have all required fields
- ✅ Generation instructions have all required fields
- ✅ Grounding requirements are present
- ✅ JSON structure is valid
- ✅ Instructions are consistent across instances

**Test coverage:** 17 tests covering all prompt methods and integration.

---

## Response Object

Prompts produce structured JSON responses. See schemas:

**Decomposition Output:**
```json
{
  "understanding": "Brief summary of what user wants",
  "requires_visualization": true,
  "libraries_needed": ["vtk", "pandas"],
  "data_files": ["example.csv"],
  "steps": [
    {
      "step_number": 1,
      "description": "Read example.csv with pandas",
      "search_query": "pandas read CSV file",
      "focus": "data_io"
    }
  ]
}
```

**Generation Output:**
```json
{
  "step_number": 1,
  "understanding": "What this step accomplishes",
  "imports": ["from vtkmodules.vtkFiltersSources import vtkCylinderSource"],
  "code": "cylinder = vtkCylinderSource()\ncylinder.SetRadius(1.0)",
  "citations": [1, 2]
}
```

For complete schema details, see:
- `llm-generation/SCHEMAS.md` - Full schema documentation
- `data_exchange.md` - Complete JSON examples

---

## Dependencies

```bash
# Currently no external dependencies
pip install -r grounding-prompting/requirements.txt
```

**Note:** Prompts are just dictionaries - no special libraries needed.

---

## Best Practices

### 1. Always Use Centralized Prompts
```python
# ✅ GOOD - Use centralized prompts
from prompt_templates import VTKPromptTemplate
template = VTKPromptTemplate()
instructions = template.get_generation_instructions()

# ❌ BAD - Don't hardcode prompts
instructions = {"task": "Generate code", "requirements": [...]}
```

### 2. Don't Modify Prompts Inline
```python
# ❌ BAD - Don't modify returned prompts
instructions = template.get_generation_instructions()
instructions["requirements"].append("Extra requirement")

# ✅ GOOD - Modify in prompt_templates.py, then use
instructions = template.get_generation_instructions()
```

### 3. Test After Modifications
```bash
# Always run tests after modifying prompts
cd tests/grounding-prompting && python test_prompt_templates.py
```

### 4. Preserve Grounding
When modifying prompts, **always keep grounding requirements** (lines 92-96). These prevent hallucination.

---

## See Also

- `llm-generation/sequential_pipeline.py` - Uses these prompts
- `llm-generation/SCHEMAS.md` - JSON schema documentation
- `example_usage.py` - Usage demonstrations
- `tests/grounding-prompting/` - Unit tests
- Top-level `data_exchange.md` - Complete JSON examples
