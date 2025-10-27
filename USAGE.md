# VTK RAG Query System - Usage Guide

The unified `query.py` script provides a simple interface to query the VTK RAG system for code generation, API documentation, and concept explanations.

## Quick Start

```bash
# Basic code generation query
python query.py "How do I create a cylinder in VTK?"

# API documentation lookup
python query.py "What methods does vtkPolyDataMapper have?"

# With visual validation (requires Docker)
python query.py "Create a red sphere" --visual-test
```

---

## Prerequisites

Before using `query.py`, ensure you have:

1. **Environment configured** (`.env` file with LLM API key)
2. **Qdrant running** (`docker run -p 6333:6333 qdrant/qdrant`)
3. **Index built** (`python build-indexes/build_qdrant_index.py`)
4. **Docker** (optional, only for `--visual-test` flag)

The script will check prerequisites and report any issues.

---

## Command-Line Options

```bash
python query.py <query> [options]
```

### Required Arguments

- `query` - Your VTK question (in quotes if it contains spaces)

### Optional Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--visual-test` | Execute generated code in Docker sandbox | `python query.py "Create sphere" --visual-test` |
| `--enrich` | Use LLM to enhance explanations | `python query.py "Create pipeline" --enrich` |
| `--output FILE` | Save JSON response to file | `python query.py "Create cone" -o result.json` |
| `--quiet` | Minimal output (only show result) | `python query.py "Create actor" --quiet` |
| `--help` | Show help message | `python query.py --help` |

---

## Query Types

The system automatically detects and routes your query to the appropriate handler:

### 1. Code Generation

**When to use:** You want VTK code to accomplish a task

**Example queries:**
```bash
python query.py "How do I create a cylinder in VTK?"
python query.py "Show me how to read an STL file and visualize it"
python query.py "Create a red sphere with a blue background"
```

**Output includes:**
- ✅ Complete executable Python code
- ✅ Step-by-step explanation
- ✅ Citations to VTK examples/docs
- ✅ Data files needed (if applicable)
- ✅ Image URL (if example has visual output)

### 2. API Documentation

**When to use:** You want to know about a specific VTK class or method

**Example queries:**
```bash
python query.py "What methods does vtkPolyDataMapper have?"
python query.py "Tell me about vtkActor class"
python query.py "What are the parameters for vtkCylinderSource?"
```

**Output includes:**
- ✅ Class name and description
- ✅ Available methods
- ✅ Usage examples
- ✅ Citations to API docs

### 3. Concept Explanations

**When to use:** You want to understand VTK concepts

**Example queries:**
```bash
python query.py "What is a mapper in VTK?"
python query.py "Explain the VTK visualization pipeline"
python query.py "What's the difference between actors and mappers?"
```

**Output includes:**
- ✅ Detailed explanation
- ✅ Related concepts
- ✅ Citations to documentation

### 4. Data File Finding

**When to use:** You want to find example data files

**Example queries:**
```bash
python query.py "What data files work with vtkSTLReader?"
python query.py "Find examples that use DICOM files"
```

**Output includes:**
- ✅ List of data files
- ✅ Download URLs
- ✅ Example code using those files

---

## Visual Testing

The `--visual-test` flag executes generated code in a Docker sandbox to verify it runs correctly and produces visual output.

**Requirements:**
- Docker must be running
- Visual testing Docker image must be built (automatic on first run)

**What it does:**
1. Executes code in isolated Docker container
2. Captures execution time and success/failure
3. Detects if visual output was generated
4. Reports any runtime errors

**Example:**
```bash
python query.py "Create a red sphere" --visual-test
```

**Output includes:**
```
🖼️  VISUAL VALIDATION:
--------------------------------------------------------------------------------
  Execution: ✅ SUCCESS
  Time: 1.23s
  Visual Output: ✅ Yes
```

**When to use:**
- ✅ Verify code executes without errors
- ✅ Confirm visualization produces output
- ✅ Test code before using in production
- ✅ Debugging execution issues

---

## LLM Enrichment

The `--enrich` flag uses the LLM to enhance code explanations with additional context.

**What it does:**
1. Takes the generated explanation
2. Enriches it with key concepts, common pitfalls, and best practices
3. Adds structured sections

**Example:**
```bash
python query.py "Create a visualization pipeline" --enrich
```

**When to use:**
- ✅ Learning VTK concepts
- ✅ Need detailed explanations
- ✅ Want best practices and tips
- ❌ Don't use if you want concise output

**Trade-offs:**
- ✅ More detailed explanations
- ❌ Slower (extra LLM call)
- ❌ May incur additional API costs

---

## Output Formats

### Console Output (Default)

Formatted, human-readable output with sections:

```
================================================================================
RESPONSE: DIRECT (CODE)
================================================================================

📝 CODE:
--------------------------------------------------------------------------------
```python
import vtk

source = vtk.vtkCylinderSource()
source.SetHeight(3.0)
source.SetRadius(1.0)
source.Update()
```

📖 EXPLANATION:
--------------------------------------------------------------------------------
This code creates a cylinder using vtkCylinderSource...

📁 DATA FILES:
--------------------------------------------------------------------------------
  • example.vtp
    URL: https://example.com/example.vtp

🔗 CITATIONS:
--------------------------------------------------------------------------------
  • CylinderExample_code_0
  • vtkCylinderSource_api
```

### JSON Output

Use `--output` flag to save structured JSON response:

```bash
python query.py "Create cylinder" --output result.json
```

**JSON structure:**
```json
{
  "response_type": "direct",
  "content_type": "code",
  "code": "import vtk\n...",
  "explanation": "This code creates...",
  "citations": ["CylinderExample_0"],
  "data_files": [
    {
      "filename": "example.vtp",
      "url": "https://example.com/example.vtp"
    }
  ],
  "image_url": "https://example.com/cylinder.png"
}
```

**When to use JSON output:**
- ✅ Integrating with other tools/scripts
- ✅ Batch processing multiple queries
- ✅ Storing results for later analysis
- ✅ Building a UI on top of the RAG system

---

## Examples

### Example 1: Basic Code Generation

```bash
python query.py "How do I create a red sphere?"
```

**Output:**
- Python code to create a sphere
- Explanation of the visualization pipeline
- Citations to relevant examples

### Example 2: Code with Visual Testing

```bash
python query.py "Create a cone with a blue background" --visual-test
```

**Output:**
- Code for cone visualization
- Execution results showing success
- Confirmation of visual output

### Example 3: API Lookup

```bash
python query.py "What methods does vtkActor have?"
```

**Output:**
- List of vtkActor methods
- Brief descriptions
- Citations to API documentation

### Example 4: Save to File

```bash
python query.py "Show me a cylinder example" --output cylinder.json
```

**Result:**
- Console shows formatted output
- `cylinder.json` contains complete JSON response
- Can be loaded and processed later

### Example 5: Batch Processing

```bash
# Create a script to query multiple things
for query in \
  "Create a sphere" \
  "Create a cone" \
  "Create a cylinder"
do
  python query.py "$query" --output "${query// /_}.json" --quiet
done
```

### Example 6: Enhanced Explanations

```bash
python query.py "Explain the VTK visualization pipeline" --enrich
```

**Output:**
- Detailed explanation with key concepts
- Common pitfalls and best practices
- Structured learning content

---

## Tips & Best Practices

### Query Writing

✅ **Do:**
- Be specific: "Create a red sphere with radius 2.0"
- Mention file types: "Read an STL file and visualize it"
- Specify details: "Create a cylinder with height 3.0 and radius 1.0"

❌ **Don't:**
- Be too vague: "Make something"
- Use ambiguous terms without context
- Assume implicit requirements

### Performance

- **Visual testing adds ~2-5s** execution time
- **Enrichment adds ~3-10s** (extra LLM call)
- **Use `--quiet`** for faster output in scripts
- **Cache JSON results** to avoid re-querying

### Troubleshooting

**"Qdrant not running"**
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

**"vtk_docs collection missing"**
```bash
python build-indexes/build_qdrant_index.py
```

**"Docker not running" (with --visual-test)**
```bash
# Start Docker Desktop or:
sudo systemctl start docker  # Linux
```

**API rate limits / costs**
- Use `--quiet` to reduce output processing
- Avoid `--enrich` if not needed
- Cache results locally with `--output`

---

## Integration Examples

### Python Script Integration

```python
import subprocess
import json

# Run query and get JSON
result = subprocess.run(
    ['python', 'query.py', 'Create a sphere', '--output', 'result.json', '--quiet'],
    capture_output=True
)

# Load result
with open('result.json') as f:
    response = json.load(f)

# Use the code
code = response['code']
exec(code)  # Execute VTK code
```

### Shell Script Integration

```bash
#!/bin/bash
# Generate code for multiple visualizations

queries=(
  "Create a red sphere"
  "Create a blue cone"
  "Create a yellow cylinder"
)

for i in "${!queries[@]}"; do
  python query.py "${queries[$i]}" \
    --visual-test \
    --output "result_$i.json" \
    --quiet
  
  if [ $? -eq 0 ]; then
    echo "✓ Query $i completed"
  else
    echo "✗ Query $i failed"
  fi
done
```

---

## Advanced Usage

### Custom Validation

Combine with your own validation:

```bash
# Generate code
python query.py "Create sphere" --output sphere.json

# Extract and validate
code=$(jq -r '.code' sphere.json)
echo "$code" | python -m py_compile  # Check syntax

# Run with your own tests
python -c "$code" && echo "Success!"
```

### CI/CD Integration

```yaml
# .github/workflows/test-vtk-rag.yml
name: Test VTK RAG

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Start Qdrant
        run: docker run -d -p 6333:6333 qdrant/qdrant
      - name: Run queries
        run: |
          python query.py "Create sphere" --visual-test
          python query.py "Create cone" --visual-test
```

---

## FAQ

**Q: Can I query multiple things at once?**  
A: Not directly, but you can script it (see batch processing example above)

**Q: How do I know if visual testing is working?**  
A: Run with `--visual-test` and check for "VISUAL VALIDATION" section in output

**Q: Can I use this without an internet connection?**  
A: No, it requires LLM API access (OpenAI, Anthropic, etc.)

**Q: How much does each query cost?**  
A: Varies by LLM provider. Typically $0.001-0.01 per query. Add ~2x for `--enrich`

**Q: Can I query local/custom VTK documentation?**  
A: Yes, but you need to rebuild the index with your custom docs

**Q: Is the generated code production-ready?**  
A: Code passes API validation and security checks. Use `--visual-test` to verify execution.

---

## See Also

- [Main README](README.md) - Project overview
- [Evaluation Guide](evaluation/README.md) - Testing and metrics
- [API Validation](api-mcp/README.md) - VTK API hallucination detection
- [Visual Testing](visual_testing/README.md) - Docker sandbox execution
