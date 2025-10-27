## Post-Processing for VTK RAG

Parse and structure LLM responses into actionable components for UI rendering, validation, and interactive flows.

## Features

✅ **Response Type Detection**
- Complete answers (with code/explanations)
- Clarifying questions (interactive flow)
- Refusals (insufficient context)

✅ **Component Extraction**
- Python code blocks (with VTK class detection)
- Explanations (separate from code)
- API references (classes and methods)
- Citations (linked to source chunks)
- Confidence levels
- Data files (with download URLs)
- Baseline images (expected outputs)

✅ **Response Enrichment**
- Generate code explanations (for CODE responses)
- Extract data files from metadata
- Link baseline images automatically
- Pass-through for API/EXPLANATION/IMAGE responses

✅ **Interactive Flow Support**
- Detect when LLM asks clarifying questions
- Extract options for user selection
- Enable multi-turn conversations

✅ **Structured Output**
- Easy access to each component
- Ready for UI rendering
- Validation-friendly

---

## File Inventory

### Core Scripts (Used in Pipeline)

| File | Purpose | Used By |
|------|---------|---------|
| `json_response_processor.py` | **Modern**: Process JSON responses from LLM (validation, enrichment, formatting) | **Current pipeline** |
| `response_enricher.py` | Enrich CODE responses with generated explanations (2nd LLM pass) | Both pipelines |

### Utility Scripts

| File | Purpose | Notes |
|------|---------|-------|
| `example_usage.py` | Demonstrate parsing and enrichment patterns | Demo script |

### Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | Dependencies (none - uses llm-generation deps) |
| `README.md` | This documentation |

---

## Quick Start

### Basic Usage

```python
from response_parser import ResponseParser, ResponseType

# Parse LLM response
parser = ResponseParser()
parsed = parser.parse(llm_response_text, metadata)

# Check response type
if parsed.response_type == ResponseType.ANSWER:
    # Extract components
    code = parsed.get_main_code()
    explanations = parsed.explanations
    citations = parsed.citations
    
elif parsed.response_type == ResponseType.CLARIFYING_QUESTION:
    # Interactive flow
    question = parsed.question
    options = parsed.options
    # Present to user, then re-query
```

### Integration with RAG Pipeline

```python
import sys
sys.path.append('llm-generation')
sys.path.append('post-processing')

from generator import VTKRAGGenerator
from response_parser import ResponseParser
from response_enricher import ResponseEnricher

# Generate response
generator = VTKRAGGenerator()
response = generator.generate(prompt, metadata)

# Post-process
parser = ResponseParser()
parsed = parser.parse(response.answer, metadata)

# Enrich based on content type
enricher = ResponseEnricher()
enriched = enricher.enrich(
    parsed=parsed,
    content_type='code',  # or 'api', 'explanation', 'image'
    metadata=metadata
)

# Access enriched components
sections = enriched.get_display_sections()
if sections.get('code'):
    code_editor.set_text(sections['code'])
if sections.get('explanation'):
    text_panel.set_text(sections['explanation'])
if sections.get('data_files'):
    data_panel.set_markdown(sections['data_files'])
if sections.get('baseline_image'):
    image_panel.set_markdown(sections['baseline_image'])
```

---

## Modern JSON-Based Processing

### `json_response_processor.py`

**The current post-processing system** for JSON responses from the LLM.

#### **Overview**

**✨ LLM Enrichment Available:**

The system can optionally improve code explanations using a second LLM pass:

**When to use:**
- Explanation is missing entirely
- Explanation is too brief (< 50 chars)
- User requests detailed explanation

**What it does:**
- Generates detailed explanations for code
- Explains each VTK class and its role
- Describes data flow through pipeline
- Highlights important method calls
- Adds educational context

**How to use:**
```python
processor = JSONResponseProcessor()

# Enrich code response with better explanation
enriched = processor.enrich_with_llm(
    response=response,
    documentation_chunks=retrieved_chunks,  # For accuracy
    llm_client=llm_client  # Optional, creates new if not provided
)

# Access enriched content
print(enriched['explanation'])  # Improved explanation
print(enriched['_enrichment']['key_points'])  # Key takeaways
print(enriched['_enrichment']['original_explanation'])  # Original preserved
```

**Performance:**
- ⚡ **Optional** - Only call when needed
- 🐌 **Adds 1-3 seconds** - Requires LLM call
- 💰 **~800 tokens** - ~$0.02 per enrichment (GPT-4)
- 💡 **Best practice** - Cache enriched responses to avoid re-enriching same code

**Example:**

```python
# Before enrichment
response = {
    "code": "cylinder = vtkCylinderSource()\nmapper = vtkPolyDataMapper()",
    "explanation": ""  # Missing!
}

# After enrichment
enriched = processor.enrich_with_llm(response, chunks)
# enriched['explanation'] = "This code creates a 3D cylinder visualization 
# using VTK's standard pipeline. vtkCylinderSource generates the cylinder 
# geometry, vtkPolyDataMapper converts it to graphics primitives..."
```

Modern VTK RAG uses structured JSON communication with the LLM:
1. LLM generates JSON responses (via `llm_client.generate_json()`)
2. JSON Response Processor validates and enriches the response
3. Returns structured `EnrichedResponse` ready for UI/evaluation

**No parsing needed** - JSON is already structured!

#### **Classes**

##### **`EnrichedResponse`** (Dataclass)

Structured response with all components extracted and validated.

```python
@dataclass
class EnrichedResponse:
    # Core content
    response_type: str          # "answer", "clarification", "error"
    content_type: str           # "code", "api", "explanation", etc.
    code: str                   # Generated code
    explanation: str            # Explanation text
    
    # Metadata
    citations: List[Dict]       # Source citations
    chunk_ids_used: List[str]  # Retrieved chunk IDs
    confidence: str             # "high", "medium", "low"
    
    # Enrichment flags
    has_code: bool              # Whether code is present
    has_explanation: bool       # Whether explanation is present
    has_citations: bool         # Whether citations are present
    
    # Optional fields
    query: str = ""             # Original query
    data_files: List[str] = field(default_factory=list)
    baseline_image: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
```

##### **`JSONResponseProcessor`**

Main processor class with validation and enrichment logic.

```python
class JSONResponseProcessor:
    """
    Process JSON responses from LLM
    
    Responsibilities:
    - Validate JSON structure
    - Extract components
    - Set enrichment flags
    - Format for UI/evaluation
    """
    
    def process(self, json_response: Dict) -> EnrichedResponse:
        """
        Process a JSON response
        
        Args:
            json_response: JSON dict from LLM
        
        Returns:
            EnrichedResponse with validated and enriched data
        """
```

#### **Usage Example**

```python
from json_response_processor import JSONResponseProcessor, EnrichedResponse

# Initialize processor
processor = JSONResponseProcessor()

# JSON response from LLM (via sequential_pipeline.process_query)
json_response = {
    "response_type": "answer",
    "content_type": "code",
    "query": "Create a cylinder",
    "code": "cylinder = vtkCylinderSource()...",
    "explanation": "This code creates...",
    "citations": [
        {"number": 1, "reason": "Step 1 documentation"}
    ],
    "chunk_ids_used": ["CylinderExample_code_0"],
    "confidence": "high"
}

# Process the response
enriched = processor.process(json_response)

# Access components
print(enriched.code)           # Generated code
print(enriched.explanation)    # Explanation text
print(enriched.has_code)       # True
print(enriched.has_citations)  # True
print(enriched.confidence)     # "high"

# Serialize for storage/transmission
enriched_dict = enriched.to_dict()
```

#### **Integration with Pipeline**

The JSON Response Processor is automatically used by `SequentialPipeline`:

```python
from sequential_pipeline import SequentialPipeline

# Pipeline already uses JSON processor internally
pipeline = SequentialPipeline(use_llm_decomposition=True)

# Returns JSON response (already processed)
response = pipeline.process_query("Create a cylinder")

# Response is a dict (not EnrichedResponse in pipeline)
# But follows same structure
print(response['code'])
print(response['explanation'])
print(response['citations'])
```

#### **Validation**

The processor validates:
- ✅ Required fields are present (`response_type`, `content_type`)
- ✅ Data types are correct (strings, lists, dicts)
- ✅ Enums are valid (`response_type` in ["answer", "clarification", "error"])
- ✅ Citations are well-formed

If validation fails, returns an error response:
```python
{
    "response_type": "error",
    "content_type": "error",
    "explanation": "Invalid response structure: missing 'code' field",
    "confidence": "low"
}
```

#### **Enrichment**

The processor adds derived fields:
- **`has_code`**: `bool(response.get('code'))`
- **`has_explanation`**: `bool(response.get('explanation'))`
- **`has_citations`**: `bool(response.get('citations'))`
- **`data_files`**: Extracted from explanation or metadata
- **`baseline_image`**: Extracted from metadata

#### **Response Types Supported**

| Type | Content Types | Fields |
|------|--------------|---------|
| `answer` | `code`, `api`, `explanation`, `image` | `code`, `explanation`, `citations` |
| `clarification` | `question` | `question`, `options`, `context` |
| `error` | `error` | `explanation` (error message) |

#### **When to Use**

✅ **Use `json_response_processor.py` for:**
- Modern pipeline with JSON output
- Structured LLM responses
- Schema-validated communication
- Current production system

❌ **Don't use for:**
- Legacy text-based responses (use `response_parser.py`)
- Custom response formats
- Non-JSON data

---

### **API Reference**

#### **`process_response(response: Dict) -> EnrichedResponse`**

Convenience function to process a JSON response.

**Args:**
- `response` (Dict): JSON response from query handler

**Returns:**
- `EnrichedResponse`: Enriched response with metadata

**Example:**
```python
enriched = process_response({
    "response_type": "answer",
    "content_type": "code",
    "code": "cylinder = vtkCylinderSource()",
    "confidence": "high",
    "citations": []
})
```

---

#### **`JSONResponseProcessor` Methods**

**`process(response: Dict) -> EnrichedResponse`**

Process and enrich a JSON response.

**Returns EnrichedResponse with fields:**
- `original`: Original JSON response
- `vtk_classes`: List of VTK classes found
- `has_code`: Boolean - has code field
- `has_citations`: Boolean - has citations
- `citation_count`: Number of citations
- `confidence`: Confidence level
- `response_type`: Type of response
- `content_type`: Content type
- `metadata`: Dict of extracted metadata

**`validate_citations(response: Dict) -> Dict`**

Validate citation structure.

**Returns:**
- `valid`: Boolean - citations valid
- `issues`: List of validation issues
- `citation_count`: Number of citations
- `citation_numbers`: List of citation numbers

**`extract_mentioned_files(response: Dict) -> Dict`**

Extract file mentions from response.

**Returns:**
- `data_files`: Explicit data files
- `code_files`: Files mentioned in code
- `mentioned_files`: All file mentions

**`summarize(response: Dict) -> str`**

Generate human-readable summary of response contents.

---

### **More Examples**

#### **Process CODE Response**

```python
response = {
    "response_type": "answer",
    "content_type": "code",
    "code": "from vtkmodules.vtkFiltersSources import vtkCylinderSource\ncylinder = vtkCylinderSource()",
    "explanation": "Creates a cylinder",
    "vtk_classes_used": ["vtkCylinderSource"],
    "citations": [{"number": 1, "reason": "cylinder example"}],
    "confidence": "high"
}

enriched = process_response(response)

print(enriched.vtk_classes)  # ['vtkCylinderSource']
print(enriched.has_code)     # True
print(enriched.confidence)   # 'high'
print(enriched.metadata['code_length'])  # 84
print(enriched.metadata['has_imports'])  # True
```

#### **Validate Citations**

```python
processor = JSONResponseProcessor()

# Good citations
response = {
    "citations": [
        {"number": 1, "reason": "First source"},
        {"number": 2, "reason": "Second source"}
    ]
}

result = processor.validate_citations(response)
print(result['valid'])           # True
print(result['citation_count'])  # 2

# Bad citations (will fail)
response = {
    "citations": [
        {"number": 1},  # Missing reason
        {"number": 1, "reason": "Duplicate"}  # Duplicate number
    ]
}

result = processor.validate_citations(response)
print(result['valid'])   # False
print(result['issues'])  # List of issues
```

#### **Extract Files**

```python
processor = JSONResponseProcessor()

response = {
    "code": "reader.SetFileName('data.csv')",
    "data_files": [{"filename": "points.csv"}]
}

files = processor.extract_mentioned_files(response)
print(files['data_files'])  # ['points.csv']
print(files['code_files'])  # ['data.csv']
```

---

### **Migration from Text Parsing**

If you were using the old `response_parser.py`:

**Old code (text parsing):**
```python
from response_parser import ResponseParser

parser = ResponseParser()
parsed = parser.parse(llm_text_response)

code_blocks = parsed.code_blocks
citations = parsed.citations
confidence = parsed.confidence
```

**New code (JSON):**
```python
from json_response_processor import process_response

# Get JSON directly from pipeline
json_response = pipeline.process_query(query)

# Process
enriched = process_response(json_response)

code = json_response['code']
citations = json_response['citations']
confidence = enriched.confidence
vtk_classes = enriched.vtk_classes  # Bonus: automatic VTK class extraction!
```

**Benefits:**
- ✅ No text parsing - direct JSON access
- ✅ Type-safe structure
- ✅ Automatic VTK class extraction
- ✅ Built-in validation
- ✅ Metadata enrichment

---

### **Comparison: Old vs New**

| Feature | Old (Text Parsing) | New (JSON Processing) |
|---------|-------------------|----------------------|
| **Input** | Unstructured text | Structured JSON |
| **Parsing** | Regex, fragile | Direct access |
| **Code extraction** | Find ```python blocks | `response['code']` |
| **Citations** | Parse [1], [2] | `response['citations']` |
| **VTK classes** | Manual regex | Automatic extraction |
| **Validation** | None | Built-in |
| **Reliability** | Breaks with format changes | Guaranteed structure |
| **Speed** | Slower (parsing) | Faster (direct access) |

---

### **Testing**

Run tests for JSON processor:

```bash
cd tests/post-processing
python test_json_response_processor.py -v
```

**18 tests covering:**
- Response processing for all content types (CODE, API, EXPLANATION, DATA)
- VTK class extraction
- Citation validation
- File extraction (data files, code files)
- Metadata enrichment
- Validation errors and edge cases

All tests passing ✅

---

### `response_enricher.py`

Response enricher for content-type specific enhancements.

**Classes:**
- `EnrichedResponse` - Enriched response with additional context
- `ResponseEnricher` - Main enricher class

**Enrichment by Content Type:**
- **CODE**: Generates explanation + extracts data files + links baseline images
- **API/EXPLANATION/IMAGE**: Pass-through (no enrichment needed)

### `example_usage.py`

Demonstrations of parser capabilities.

**Examples:**
1. Complete answer parsing
2. Clarifying question detection
3. Refusal handling
4. Integrated pipeline
5. Component extraction for different uses
6. Data files and baseline images

---

## Response Enrichment Workflow

### When to Enrich

```python
# CODE responses → Always enrich
if content_type == 'code' and parsed.has_code:
    enriched = enricher.enrich(parsed, 'code', metadata)
    # → Generates explanation
    # → Extracts data files
    # → Links baseline images

# API/EXPLANATION/IMAGE → Pass through
else:
    enriched = enricher.enrich(parsed, content_type, metadata)
    # → No enrichment, just wraps parsed response
```

### CODE Response Enrichment

For CODE responses, enrichment adds:

1. **Generated Explanation** (via LLM + API docs)
   - Extracts VTK classes from generated code
   - Retrieves API documentation for those classes (3 chunks, ~600 tokens)
   - Passes code + API docs to LLM for grounded explanation
   - Gets: What code does, VTK class descriptions, method parameters
   - **Token budget: ~1,700 tokens** (code 400t + API docs 600t + prompt 200t + response 500t)

2. **Data Files** (from metadata)
   - Extracted from all retrieved chunks
   - Required data files with download URLs
   - Formatted download commands (curl/wget)

3. **Baseline Images** (from metadata)
   - Extracted from top-ranked chunk (most relevant example)
   - Expected output image URL
   - Formatted display sections

### Example: Full Pipeline with Enrichment

```python
from retrieval_pipeline.task_specific_retriever import TaskSpecificRetriever
from llm_generation.generator import VTKRAGGenerator
from post_processing.response_parser import ResponseParser
from post_processing.response_enricher import ResponseEnricher

# 1. Retrieve CODE examples
retriever = TaskSpecificRetriever()
chunks = retriever.retrieve_code("create cylinder", top_k=3)

# 2. Generate code with LLM
generator = VTKRAGGenerator()
response = generator.generate(prompt, metadata)

# 3. Parse response
parser = ResponseParser()
parsed = parser.parse(response.answer, metadata)

# 4. Enrich CODE response
# For CODE: Retrieves API docs for VTK classes + generates explanation
# For API/EXPLANATION/IMAGE: Pass-through (no enrichment)
enricher = ResponseEnricher(retriever=retriever)  # Reuse retriever
enriched = enricher.enrich(parsed, 'code', metadata)

# 5. Display all sections
sections = enriched.get_display_sections()
print(sections['code'])           # The generated code
print(sections['explanation'])    # Explanation with API details
print(sections['data_files'])     # Download commands (if needed)
print(sections['baseline_image']) # Expected output image (if available)

# Enrichment process for CODE:
# - Extracts VTK classes from code (e.g., vtkCylinderSource, vtkActor)
# - Retrieves API docs for those classes (~600 tokens)
# - Generates explanation using code + API docs (~1,700 tokens total)
```

---

## Response Types

### 1. Answer (`ResponseType.ANSWER`)

Complete response with code and/or explanations.

**Components:**
```python
parsed.code_blocks      # List[CodeBlock]
parsed.explanations     # List[str]
parsed.api_references   # List[APIReference]
parsed.citations        # List[Citation]
parsed.confidence       # ConfidenceLevel
parsed.data_files       # List[DataFileInfo]
parsed.baseline_images  # List[BaselineImage] ◄── NEW
```

**Example:**
```
To create a cylinder, use vtkCylinderSource [1]:

\`\`\`python
cylinder = vtkCylinderSource()
cylinder.SetRadius(5.0)
\`\`\`

This creates a cylinder with radius 5 [1].
```

### 2. Clarifying Question (`ResponseType.CLARIFYING_QUESTION`)

LLM needs more information before answering.

**Components:**
```python
parsed.question         # str
parsed.options          # List[str]
parsed.is_interactive   # bool = True
```

**Example:**
```
Which type of visualization do you need?
1. Basic rendering
2. Custom colors
3. Animation
```

### 3. Refusal (`ResponseType.REFUSAL`)

Cannot answer due to insufficient context.

**Components:**
```python
parsed.refusal_reason   # str
```

**Example:**
```
I don't have enough information in the provided documentation 
to answer this question accurately.
```

---

## Parsed Components

### Code Blocks

```python
@dataclass
class CodeBlock:
    code: str              # Source code
    language: str          # "python"
    has_imports: bool      # Contains import statements
    vtk_classes: List[str] # VTK classes used
```

**Usage:**
```python
for block in parsed.code_blocks:
    print(f"VTK classes: {block.vtk_classes}")
    if block.has_imports:
        print("Ready to run")
    
# Get main code block
main_code = parsed.get_main_code()
```

### Citations

```python
@dataclass
class Citation:
    number: int            # [1], [2], etc.
    chunk_id: str          # Source chunk ID
    source_type: str       # api_doc, example, test
```

**Usage:**
```python
for citation in parsed.citations:
    print(f"[{citation.number}] {citation.chunk_id}")
    
# Link to source chunks
metadata['chunk_details'][citation.number - 1]
```

### API References

```python
@dataclass
class APIReference:
    class_name: str        # vtkCylinderSource
    methods: List[str]     # ['SetRadius', 'SetHeight']
```

**Usage:**
```python
for api in parsed.api_references:
    print(f"{api.class_name}: {', '.join(api.methods)}")
    
# All VTK classes mentioned
all_classes = parsed.get_all_vtk_classes()
```

### Confidence Levels

```python
class ConfidenceLevel(Enum):
    HIGH = "high"          # Very confident
    MEDIUM = "medium"      # Somewhat confident
    LOW = "low"            # Low confidence
    UNCERTAIN = "uncertain" # Explicitly uncertain
```

**Detected from language:**
- HIGH: "definitely", "certainly", "clearly"
- LOW: "might", "possibly", "one approach"
- UNCERTAIN: "not sure", "cannot guarantee"

### Data Files ◄── NEW

```python
@dataclass
class DataFileInfo:
    filename: str              # e.g., "Data/beach.jpg" or "mug.e"
    download_urls: List[Dict]  # Download URLs and methods
    sha512_url: str            # SHA512 file URL (for tests)
    file_type: str             # "test" or "example"
```

**Usage:**
```python
# Check if response requires data files
if parsed.has_data_files():
    # Format download section
    data_section = parsed.format_data_section(style="markdown")
    print(data_section)
    
# Access individual files
for data_file in parsed.data_files:
    print(f"Required: {data_file.filename}")
    for url_info in data_file.download_urls:
        print(f"  Download: {url_info['url']}")
```

**Automatic extraction:**
Data files are automatically extracted from retrieved chunk metadata when it includes `data_download_info` fields (populated by augmentation in source repositories).

**Download formatting:**
The parser formats download sections with direct `curl` and `wget` commands, plus the direct URL for manual download. No additional utilities required.

### Baseline Images ◄── NEW

```python
@dataclass
class BaselineImage:
    image_url: str             # Direct URL to baseline image
    local_image_path: str      # Local path to image (if available)
    has_baseline: bool         # Whether baseline exists
    source_type: str           # "test" or "example"
```

**Usage:**
```python
# Check if response has baseline images
if parsed.has_baseline_images():
    # Get first available baseline
    baseline = parsed.get_baseline_image()
    print(f"Expected output: {baseline.image_url}")
    
    # Format baseline section
    baseline_section = parsed.format_baseline_section(style="markdown")
    print(baseline_section)

# Access all baseline images
for img in parsed.baseline_images:
    if img.is_available():
        print(f"Baseline: {img.image_url}")
```

**Automatic extraction:**
- **Examples**: Extracted from `image_url` and `local_image_path` fields in chunk metadata
  - Example URL: `https://github.com/Kitware/vtk-examples/blob/gh-pages/src/Testing/Baseline/Python/GeometricObjects/TestCylinderExample.png?raw=true`
- **Tests**: Constructed from `has_baseline` flag and `file_path`
  - Converts: `Charts/Core/Testing/Python/TestBarGraph.py` → `https://gitlab.kitware.com/vtk/vtk/-/raw/master/Testing/Data/Baseline/TestBarGraph.png`

**Example formatted output:**
```markdown
## 🖼️ Expected Output

![Expected Output](https://github.com/Kitware/vtk-examples/.../TestCylinderExample.png)

**Baseline Image URL:** https://...

*Source: example*
```

---

## Interactive Flows

### Handling Clarifying Questions

```python
def handle_response(parsed, original_query):
    if parsed.response_type == ResponseType.CLARIFYING_QUESTION:
        # Present options to user
        print(parsed.question)
        for i, option in enumerate(parsed.options, 1):
            print(f"{i}. {option}")
        
        # Get user choice
        choice = input("Your choice: ")
        
        # Re-query with clarification
        clarified_query = f"{original_query} (specifically: {parsed.options[int(choice)-1]})"
        
        # Restart pipeline with clarified query
        return run_pipeline(clarified_query)
    
    elif parsed.response_type == ResponseType.ANSWER:
        # Display answer
        return display_answer(parsed)
    
    elif parsed.response_type == ResponseType.REFUSAL:
        # Handle refusal
        return handle_refusal(parsed)
```

---

## Use Cases

### 1. UI Rendering

Separate display of code, explanations, and data files:

```python
# Display code in code editor
if parsed.has_code:
    code_editor.set_text(parsed.get_main_code())

# Display explanations in text panel
for explanation in parsed.explanations:
    text_panel.append(explanation)

# Display data file section
if parsed.has_data_files():
    data_section = parsed.format_data_section(style="markdown")
    data_panel.set_markdown(data_section)

# Display baseline image section ◄── NEW
if parsed.has_baseline_images():
    baseline_section = parsed.format_baseline_section(style="markdown")
    baseline_panel.set_markdown(baseline_section)
    # Or display image directly
    baseline_img = parsed.get_baseline_image()
    if baseline_img and baseline_img.image_url:
        image_viewer.load_url(baseline_img.image_url)

# Display citations with links
for citation in parsed.citations:
    link = create_source_link(citation)
    citation_panel.append(link)
```

### 2. Code Execution

Extract runnable code:

```python
if parsed.has_code:
    code = parsed.get_main_code()
    
    # Validate has imports
    if not parsed.code_blocks[0].has_imports:
        print("⚠️  Warning: Missing imports")
    
    # Execute
    exec(code)
```

### 3. Citation Validation

Verify citations are valid:

```python
max_citations = len(metadata['chunk_details'])

for citation in parsed.citations:
    if citation.number > max_citations:
        print(f"⚠️  Invalid citation: [{citation.number}]")
    else:
        print(f"✓ Valid: [{citation.number}] → {citation.chunk_id}")
```

### 4. Documentation Generation

Extract API usage:

```python
# Document which VTK classes are used
all_classes = parsed.get_all_vtk_classes()

for api in parsed.api_references:
    doc = f"# {api.class_name}\n"
    doc += f"Methods: {', '.join(api.methods)}\n"
    doc += f"Source: {get_source_for_class(api.class_name)}\n"
```

### 5. Confidence-Based Display

Adjust UI based on confidence:

```python
if parsed.confidence == ConfidenceLevel.HIGH:
    display_with_checkmark(parsed)
elif parsed.confidence == ConfidenceLevel.LOW:
    display_with_disclaimer(parsed)
elif parsed.confidence == ConfidenceLevel.UNCERTAIN:
    display_with_warning(parsed)
```

---

## Integration Points

### Stage 5.5-5.6: After LLM Generation

```
┌──────────────────────────────────────────────────┐
│  Stage 5: LLM Generation                         │
│  Input: Grounded prompt                          │
│  Output: Raw LLM response                        │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│  Stage 5.5: Parsing                              │
│  Input: Raw response + metadata                  │
│  Output: Structured ParsedResponse               │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│  Stage 5.6: Enrichment  ◄── NEW                  │
│  Input: ParsedResponse + content_type + metadata │
│  Output: EnrichedResponse                        │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
           ┌─────┴─────┐
           │ Content   │
           │  Type?    │
           └─────┬─────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
     ▼           ▼           ▼
   CODE      API/EXP     IMAGE
     │         /IMG          │
     │           │           │
     ▼           ▼           ▼
Generate    Pass        Pass
Explanation Through    Through
+ Extract   (done)      (done)
Data/Images
     │
     ▼
  Display
All Sections
```

---

## Testing

Run the test suite:

```bash
# Test parser
python post-processing/response_parser.py

# Test all examples
python post-processing/example_usage.py
```

**Test Coverage:**
- ✅ Response type detection
- ✅ Code block extraction
- ✅ Citation parsing
- ✅ API reference extraction
- ✅ Data file extraction
- ✅ Baseline image extraction
- ✅ Confidence detection

---

## Dependencies

**None!** Uses Python standard library only:
- `re` - Regular expressions
- `dataclasses` - Structured data
- `enum` - Enumerations
- `typing` - Type hints

---

## Best Practices

### 1. Always Parse Responses

```python
# ✓ GOOD: Always post-process
parsed = parser.parse(raw_response, metadata)
if parsed.has_code:
    display_code(parsed.get_main_code())

# ✗ BAD: Using raw text
display_code(raw_response)  # Includes explanations!
```

### 2. Handle All Response Types

```python
# ✓ GOOD: Handle all types
if parsed.response_type == ResponseType.ANSWER:
    display_answer(parsed)
elif parsed.response_type == ResponseType.CLARIFYING_QUESTION:
    handle_interactive(parsed)
elif parsed.response_type == ResponseType.REFUSAL:
    suggest_alternatives(parsed)

# ✗ BAD: Assuming always answer
display_answer(parsed)  # Breaks on questions!
```

### 3. Link Citations to Metadata

```python
# ✓ GOOD: Link citations
parsed = parser.parse(response, metadata)  # Pass metadata
for citation in parsed.citations:
    source = metadata['chunk_details'][citation.number - 1]
    display_source_link(citation, source)

# ✗ BAD: Citations without sources
for citation in parsed.citations:
    print(citation.number)  # No context!
```

### 4. Validate Before Execution

```python
# ✓ GOOD: Validate code
if parsed.has_code:
    main_code = parsed.get_main_code()
    vtk_classes = parsed.get_all_vtk_classes()
    
    if validate_vtk_classes(vtk_classes):
        exec(main_code)
    else:
        print("⚠️  Unknown VTK classes")

# ✗ BAD: Blind execution
exec(parsed.get_main_code())  # Risky!
```

---

## Future Enhancements

Potential additions:

1. **Code Validation**
   - Syntax checking
   - Import validation
   - VTK API version checking

2. **Enhanced Confidence Detection**
   - ML-based confidence scoring
   - Citation density analysis
   - Code complexity metrics

3. **Enhanced Data File Support** ◄── PARTIALLY COMPLETE
   - ✅ Extract from metadata
   - ✅ Format download sections
   - ⏳ Automatic file validation
   - ⏳ Size information display

4. **Response Templates**
   - Standardized format enforcement
   - Structured output from LLM
   - JSON mode for supported models

---

## License

MIT License - Part of VTK RAG project
