# VTK RAG Evaluation

Measure retrieval quality and end-to-end code generation quality using standard IR metrics and focused code quality metrics, with optional visual validation.

## Overview

This module provides comprehensive evaluation for the VTK RAG pipeline:

- ✅ **Retrieval metrics** - Recall@k, nDCG@k, MRR (IR standard)
- ✅ **Code quality metrics** - Exactness, correctness, syntax validity
- ✅ **Validation tracking** - Error rates, fix attempts, success rates
- ✅ **API validation metrics** - VTK API hallucination detection, fix rates (NEW)
- ✅ **Security metrics** - Security check coverage, pass rate, issues found
- ✅ **Visual validation** - Code execution, visual output, regression detection (optional)
- ✅ **Automated test set** - ~850 examples from VTK Python examples
- ✅ **Detailed analysis** - Automatic summary reports with insights

---

## Quick Start

### 1. Build Test Set

```bash
# Build and augment test set (combines building + augmentation)
python evaluation/test_set_builder.py
```

Creates `data/processed/test_set.jsonl` with ~850 examples including:
- Queries, gold code, gold explanations
- Step-by-step decomposition (deterministic)
- Per-step retrieval results

### 2. Evaluate Retrieval (Fast, Free)

Tests retrieval quality by comparing retrieved chunks against ground truth.

```bash
python evaluation/evaluator.py \
    --test-set data/processed/test_set.jsonl \
    --mode retrieval \
    --num-examples 20 \
    --output evaluation/eval_retrieval.json
```

**Output:**
- `eval_retrieval.json` - Raw metrics (recall@k, nDCG@k, MRR)
- `eval_retrieval_summary.txt` - Human-readable analysis

**Metrics included:** Recall@k, nDCG@k, MRR

### 3. Evaluate End-to-End (Calls LLM)

Tests the full pipeline by generating code and measuring output quality.

```bash
python evaluation/evaluator.py \
    --test-set data/processed/test_set.jsonl \
    --mode end-to-end \
    --num-examples 10 \
    --output evaluation/eval_e2e.json
```

**Output:**
- `eval_e2e.json` - Raw metrics (code quality, validation)
- `eval_e2e_summary.txt` - Human-readable analysis

**Metrics included:** Code exactness, code correctness, syntax validity, validation results, API validation, security validation

⚠️ **Warning:** This calls the LLM and may incur costs!

### 4. Evaluate with Visual Testing (Optional)

Execute generated code in Docker and validate visual output.

```bash
# Requires Docker image
cd visual_testing && docker build -t vtk-visual-testing .

# Run evaluation with visual testing
RUN_VISUAL_TESTS=1 python evaluation/evaluator.py \
    --test-set data/processed/test_set.jsonl \
    --mode end-to-end \
    --num-examples 5 \
    --enable-visual-testing \
    --output evaluation/eval_e2e_visual.json
```

**Additional metrics:**
- Execution success rate
- Visual output generation rate  
- Visual regression detection (vs. baselines)
- SSIM similarity scores
- Execution times

**Create baselines on first run:**
```bash
CREATE_BASELINES=1 RUN_VISUAL_TESTS=1 python evaluation/evaluator.py \
    --mode end-to-end \
    --enable-visual-testing \
    --num-examples 10
```

---

## File Inventory

### Core Scripts

| File | Purpose | Notes |
|------|---------|-------|
| `evaluator.py` | Main evaluation orchestrator | Runs retrieval or e2e evaluation |
| `visual_evaluator.py` | Visual validation evaluator | Executes code in Docker, compares output |
| `retrieval_metrics.py` | Retrieval metrics (Recall, nDCG, MRR) | IR standard metrics |
| `end_to_end_metrics.py` | Code quality metrics | Exactness, correctness, validation |

### Test Set Management

| File | Purpose | Notes |
|------|---------|-------|
| `test_set_builder.py` | Build and augment test set | Creates complete test_set.jsonl with steps + retrieval |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | This documentation |
| `requirements.txt` | No external dependencies (pure Python) |

---

## Evaluation Modes

### Mode Separation

The evaluator uses **two distinct modes** with different metrics:

| Mode | Purpose | Metrics | What It Measures |
|------|---------|---------|------------------|
| `--mode retrieval` | Test retrieval quality | Recall@k, nDCG@k, MRR | Does retrieval find the right chunks? |
| `--mode end-to-end` | Test full pipeline | Code quality, validation | Does the pipeline produce good code? 

---

## Metrics

### Retrieval Metrics (IR Standard)

**Used in:** `--mode retrieval` only

#### **Recall@k** - Fraction of relevant chunks found in top-k
```
Recall@k = (# relevant in top-k) / (# total relevant)
```
- **Range:** [0.0, 1.0]
- **Higher is better**
- **Target @10:** >0.85
- **What it measures:** Does the system find the relevant documentation chunks?

#### **nDCG@k** - Normalized Discounted Cumulative Gain (ranking quality)
```
DCG@k = Σ relevance / log2(position + 1)
nDCG@k = DCG@k / IDCG@k
```
- **Range:** [0.0, 1.0]
- **Higher is better**
- **Target @10:** >0.95
- **What it measures:** Are relevant chunks ranked highly? Rewards chunks at top positions.

#### **MRR** - Mean Reciprocal Rank (first relevant position)
```
RR = 1 / rank_of_first_relevant
MRR = average(RR) across queries
```
- **Range:** [0.0, 1.0]
- **Higher is better**
- **Target:** ~1.0
- **What it measures:** How quickly does the first relevant chunk appear?

---

### End-to-End Metrics (Code Quality)

**Used in:** `--mode end-to-end` only

#### **Code Exactness** - Character-by-character text similarity

**⚠️ Low scores are EXPECTED and OK for pythonic code!**

- **Range:** [0.0, 1.0]
- **How measured:** 
  1. Normalizes code (removes comments, docstrings, extra whitespace)
  2. Compares character-by-character using `SequenceMatcher`
  3. Returns similarity ratio
- **What it measures:** How similar is the generated code text to the gold reference?
- **Target:** N/A - use Correctness instead

**Why exactness is often low (20-40%):**
- ✅ **Different but equivalent:** Pythonic API (`vtkActor(mapper=x)`) vs traditional API (`actor.SetMapper(x)`)
- ✅ **Concise vs verbose:** Generated code omits comments, styling, optional methods
- ✅ **Variable names:** `mapper` vs `cylinderMapper` vs `polydata_mapper`
- ✅ **Structure:** No `def main()` wrapper, different formatting

**💡 Recommendation:** Focus on **Correctness** instead - it measures functional equivalence, not text similarity.

---

#### **Code Correctness** - Functional component completeness

**This is the primary quality metric!**

- **Range:** [0.0, 1.0]
- **Target:** >0.70 (good), >0.80 (great)
- **How measured:** Average of 3 checks:

**Check 1: VTK Classes (33%)**
```
Score = (# required classes present) / (# required classes in gold)
```
- Checks if generated code uses the same VTK classes as gold
- Example: If gold uses `vtkCylinderSource`, `vtkPolyDataMapper`, `vtkActor`, generated must have them too

**Check 2: Imports (33%)**
```
Score = 1.0 if has imports, 0.0 otherwise
```
- Verifies code has proper import statements

**Check 3: Method Calls - **Pythonic API Aware** (33%)**
```
Score = (# methods matched) / (# methods in gold)
```
- Compares method calls between generated and gold code
- **Automatically credits pythonic API equivalents:**
  - ✅ `vtkActor(mapper=x)` counts as `.SetMapper()`
  - ✅ `obj.render_window = x` counts as `.SetRenderWindow()`
  - ✅ `source >> mapper` counts as `.SetInputConnection()` and `.GetOutputPort()`
- Missing optional methods (styling, camera control) reduce score but code still works

**Example Breakdown:**
```
Generated: 6/7 classes (85%) + imports (100%) + 9/21 methods (43%)
Correctness = (85 + 100 + 43) / 3 = 76%
```

**What counts as "missing" methods:**
- ❌ **Required:** Core pipeline methods (SetInputConnection, AddActor, Render, Start)
- ⚠️ **Optional:** Styling (SetColor, Rotate), window config (SetSize, SetWindowName), camera (Zoom, ResetCamera)

---

#### **Explanation Quality** - Explanation-to-code ratio

- **Range:** [0.0, 1.0] mapped to quality categories
- **Target:** >0.70 (good explanation)
- **How measured:** Compares explanation length to code length

**Scoring tiers:**
- **<1x code length:** Poor (60%) - Explanation shorter than code
- **1-2x:** OK (70%) - Basic explanation
- **2-3x:** Good (80%) - Detailed explanation
- **3-4x:** Great (90%) - Comprehensive explanation  
- **4-5x+:** Perfect (100%) - Extensive explanation

**Example:**
```
Code: 800 characters
Explanation: 2,400 characters
Ratio: 3x → Great (90%)
```

**What it measures:** Does the generated response provide adequate context and understanding, not just code?

---

#### **Code Syntax Valid** - Python syntax correctness

- **Range:** [0.0, 1.0]
- **Target:** ~1.0 (100%)
- **How measured:** Percentage of examples with valid Python syntax
- **What it measures:** Does the code parse without syntax errors?

---

#### **Validation Metrics** - Error detection and fixing

Only applicable if validation is enabled (`--enable-validation`).

- **`validation_attempted`** - % of examples that went through validation
- **`validation_errors_found`** - % with syntax/import errors detected
- **`validation_needed_retry`** - % that needed automatic fixes
- **`validation_success_rate`** - % that passed validation after fixes
- **`validation_avg_retries`** - Average number of fix attempts per example

**What they measure:** How often does validation catch issues, and can they be auto-fixed?

---

#### **Security Validation Metrics** - Code safety checks

**Always enabled** for all generated code.

- **`security_check_performed`** - % of examples that went through security checks
  - **Target:** 100% (all code should be checked)
- **`security_check_passed`** - % of checked examples that passed
  - **Target:** >95% (most code should be safe)
- **`security_issues_found`** - % of checked examples with security issues
  - **Target:** <5% (few dangerous patterns)

**What they measure:**
- Are we checking all code for security?
- Is the LLM generating safe code?
- What percentage has dangerous patterns?

**What's checked:**
- ✅ Allows: VTK I/O operations, pandas/numpy
- ❌ Blocks: Direct Python file I/O, system operations, network access, code execution

**Example good results:**
```
Security Validation Metrics:
  Checked:       100.0% (all code validated)
  Passed:        100.0% (no dangerous patterns)
  Issues Found:  0.0% (clean code)
```

**Warning signs:**
- `Checked < 100%` → Security validator not running on all code (bug)
- `Passed < 95%` → LLM frequently generates unsafe patterns (prompt issue)
- `Issues > 5%` → Too many dangerous patterns detected (model tuning needed)

---

#### **API Validation Metrics** - VTK API Hallucination Detection (NEW)

**Enabled by default** when `enable_api_validation=True` (default in pipeline).

- **`api_validation_attempted`** - % of examples that went through API validation
  - **Target:** 100% (all code should be validated)
- **`api_errors_detected`** - % of examples with VTK API errors (hallucinations)
  - **Expected:** 5-15% (LLMs occasionally hallucinate methods/classes)
- **`api_errors_fixed`** - % of detected errors successfully fixed by LLM
  - **Target:** >90% (most errors should be fixable)
- **`api_errors_unfixed`** - % of detected errors that remain after LLM repair
  - **Target:** <10% (few errors remain unfixed)
- **`api_avg_errors_per_query`** - Average API errors per query
  - **Target:** <0.2 (mostly error-free)

**What they measure:**
- How often does the LLM hallucinate VTK APIs?
- Can the LLM fix API errors when provided with feedback?
- What types of errors occur most frequently?

**Types of errors detected:**
- ✅ **Non-existent methods** - e.g., `SetOutputWholeExtent()` on `vtkImageStencilToImage`
- ✅ **Non-existent classes** - e.g., `vtkImageDataToPolyDataConverter`
- ✅ **Wrong import modules** - e.g., importing `vtkPolyDataMapper` from wrong package

**Example good results:**
```
API Validation Metrics (VTK API Hallucination Detection):
  Attempted:     100.0% (all code validated)
  Errors Found:  10.0% (1/10 queries with hallucinations)
  Errors Fixed:  100.0% (LLM fixed all errors)
  Errors Unfixed:0.0% (no remaining errors)
  Avg Errors:    0.10 (per query)
```

**Validation Flow:**
1. **Detect:** API validator checks generated code against VTK API index (2,942 classes)
2. **Report:** Errors passed to LLM validator with suggestions
3. **Repair:** LLM fixes API errors + other issues in single pass
4. **Re-validate:** Confirm API errors are resolved

**Performance:**
- Detection: <5ms per query
- Index load: <1 second (one-time at startup)
- Fix rate: ~95% (LLM successfully repairs most hallucinations)

---

### Visual Validation Metrics (Optional)

**Used in:** `--mode end-to-end --enable-visual-testing` only

Requires Docker and `RUN_VISUAL_TESTS=1` environment variable.

#### **Execution Success** - Does generated code run without errors?

- **execution_attempted** - % of examples that attempted execution
  - **Target:** 100% (when visual testing enabled)
- **execution_success** - % of attempted executions that succeeded
  - **Target:** >80% (most code should run)
- **execution_failed** - % of attempted executions that failed
  - **Target:** <20% (few runtime errors)
- **avg_execution_time** - Average execution time in seconds
  - **Target:** <5s (fast execution)

**What it measures:**
- Can the generated code actually execute?
- Are there runtime errors, missing imports, or invalid API calls?

#### **Visual Output** - Does code produce rendered output?

- **has_visual_output** - % of successful executions that captured output
  - **Target:** >90% (for rendering code)
- **missing_visual_output** - % that didn't produce output
  - **Target:** <10% (few missing renders)
- **avg_output_size_kb** - Average PNG size in kilobytes
  - **Typical:** 5-50 KB

**What it measures:**
- Does rendering code actually create visual output?
- Is the render window properly configured?

#### **Visual Regression** - Does output differ from baseline?

- **with_baseline** - % of outputs with baseline comparison
  - Increases as more baselines are created
- **regression_passed** - % of baseline comparisons that passed
  - **Target:** >95% (few regressions)
- **regression_failed** - % that detected visual differences
  - **Target:** <5% (few visual changes)
- **avg_similarity** - Average SSIM similarity score (0-1)
  - **Target:** >0.95 (high visual similarity)

**What it measures:**
- Does generated code produce the expected visual output?
- Are there rendering regressions compared to known-good outputs?

**Example results:**
```
VISUAL VALIDATION METRICS
────────────────────────────────────────────────────────────────

Execution:
  Attempted:     100.0%
  Success:       85.0%
  Failed:        15.0%
  Avg Time:      1.23s

Visual Output:
  Has Output:    88.2%
  Missing:       11.8%
  Avg Size:      5.2 KB

Visual Regression:
  With Baseline: 70.6%
  Passed:        95.0%
  Failed:        5.0%
  Avg Similarity:0.978

Overall:
  Validation Passed: 75.0%
```

---

## Usage

### Retrieval Only Evaluation

```python
from evaluator import RAGEvaluator
from pathlib import Path

# Initialize (no LLM needed)
evaluator = RAGEvaluator(enable_llm=False)

# Load test set
test_set = evaluator.load_test_set(
    Path('data/processed/test_set.jsonl')
)

# Evaluate retrieval
results = evaluator.evaluate_retrieval_only(test_set[:50])

# Generate report
evaluator.generate_report(
    results,
    Path('evaluation/my_eval.json'),
    mode='retrieval'
)
```

**Output files:**
- `my_eval.json` - Raw metrics data
- `my_eval_summary.txt` - Analysis with insights

### End-to-End Evaluation

```python
# Initialize with LLM (requires .env with API key)
evaluator = RAGEvaluator(
    enable_llm=True,
    enable_validation=True
)

# Load test set
test_set = evaluator.load_test_set(
    Path('data/processed/test_set.jsonl')
)

# Evaluate complete pipeline
results = evaluator.evaluate_end_to_end(test_set[:10])

# Generate report
evaluator.generate_report(
    results,
    Path('evaluation/eval_e2e.json'),
    mode='end-to-end'
)
```

### End-to-End with Visual Testing

```python
import os

# Enable visual testing
os.environ['RUN_VISUAL_TESTS'] = '1'
# Optional: create baselines for missing tests
# os.environ['CREATE_BASELINES'] = '1'

# Initialize with visual testing
evaluator = RAGEvaluator(
    enable_llm=True,
    enable_validation=True,
    enable_visual_testing=True  # NEW
)

# Load test set
test_set = evaluator.load_test_set(
    Path('data/processed/test_set.jsonl')
)

# Evaluate with visual validation
results = evaluator.evaluate_end_to_end(test_set[:5])

# Generate report (includes visual metrics)
evaluator.generate_report(
    results,
    Path('evaluation/eval_e2e_visual.json'),
    mode='end-to-end'
)

# Cleanup
evaluator.visual_evaluator.cleanup()
```

**Requirements:**
- Docker installed and running
- `vtk-visual-testing` image built (`cd visual_testing && docker build -t vtk-visual-testing .`)
- `RUN_VISUAL_TESTS=1` environment variable

---

## Example Output

### Retrieval Evaluation Summary

```
================================================================================
EVALUATION REPORT
================================================================================

Mode: retrieval
Examples: 847

--------------------------------------------------------------------------------
RETRIEVAL METRICS (Query-Level Aggregate)
--------------------------------------------------------------------------------
Recall@1:  0.094
Recall@3:  0.282
Recall@5:  0.471
Recall@10: 0.892

nDCG@3:    1.000
nDCG@5:    1.000
nDCG@10:   1.000

MRR:       1.000

--------------------------------------------------------------------------------
DETAILED RETRIEVAL ANALYSIS
--------------------------------------------------------------------------------

Recall@N Progression:
  Recall@ 1: 0.094 (9.4%)
  Recall@ 3: 0.282 (28.2%)
  Recall@ 5: 0.471 (47.1%)
  Recall@10: 0.892 (89.2%)
  Recall@15: 0.998 (99.8%)
  Recall@20: 1.000 (100.0%)

Ground Truth Statistics:
  Avg relevant chunks per query: 10.95
  Max relevant chunks: 18
  Queries with >10 relevant: 470 (55.5%)

System Performance:
  Queries with 100% recall in top 20: 847/847 (100.0%)
  ✓ System finds ALL relevant chunks (Recall@20 ≈ 100%)
  ✓ Excellent ranking quality (nDCG@10 = 1.000)
  ℹ Recall@10 limited by ground truth size (avg 11.0 relevant > 10)
  ✓ All queries achieve ≥90% recall in top 20

================================================================================
CONCLUSION
================================================================================
✓ Retrieval system performing EXCELLENTLY
  - Finds all relevant chunks
  - Ranks them well (most in top 10)
  - Production ready!
```

### End-to-End Evaluation Summary

```
================================================================================
EVALUATION REPORT
================================================================================

Mode: end-to-end
Examples: 10

--------------------------------------------------------------------------------
END-TO-END CODE QUALITY METRICS
--------------------------------------------------------------------------------

Code Quality:
  Exactness:     31.1% (similarity to gold)
  Correctness:   80.2% (has all components)
  Syntax Valid:  100.0% (no syntax errors)

Explanation:
  Quality:       80.0%

Validation/Error Metrics:
  Attempted:     100.0%
  Errors Found:  0.0%
  Needed Retry:  0.0%
  Success Rate:  0.0%
  Avg Retries:   0.00
```

---

## Configuration

### Command-Line Options

```bash
python evaluation/evaluator.py \
    --test-set <path>           # Test set JSONL file
    --mode <retrieval|end-to-end>  # Evaluation mode
    --num-examples <N>          # Number of examples to evaluate
    --output <path>             # Output JSON file
    --no-validation             # Disable code validation (e2e only)
```

### Environment Variables

For end-to-end evaluation, configure LLM in `.env`:

```bash
# LLM Provider
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5
ANTHROPIC_TEMPERATURE=0.1

# Validation (optional)
VALIDATE_CODE=true
VALIDATION_MAX_RETRIES=2
```

See `llm-generation/README.md` for full LLM configuration options.

---

## Test Set

### Statistics

**~850 Examples** from VTK Python examples:
- Categories: GeometricObjects (250+), VisualizationAlgorithms, Rendering, IO, etc.
- Code length: 56-49,594 chars (avg: 5,123)
- Unique VTK classes: 792
- All have gold code, explanation, and supporting chunks

### Structure

Each test example contains:

```json
{
  "query": "How can I create a basic rendering of a polygonal cylinder?",
  "gold_code": "#!/usr/bin/env python...",
  "gold_explanation": "This example demonstrates...",
  "supporting_chunk_ids": ["CylinderExample_code_0", ...],
  "steps": ["Create cylinder", "Create mapper", ...],
  "step_results": [
    {
      "step_description": "Create cylinder",
      "retrieved_chunks": [...],
      "num_retrieved": 5
    }
  ],
  "metadata": {
    "title": "CylinderExample",
    "category": "GeometricObjects",
    "vtk_classes": ["vtkCylinderSource", ...],
    "has_image": true,
    "data_files": [],
    "has_baseline": true
  }
}
```

---

## Best Practices

### 1. Start with Retrieval

Always evaluate retrieval first (fast, free, no LLM):

```bash
python evaluation/evaluator.py --mode retrieval --num-examples 847
```

### 2. Small E2E Samples

Use small samples for end-to-end (expensive):

```bash
python evaluation/evaluator.py --mode end-to-end --num-examples 10
```

### 3. Track Over Time

Save with timestamps:

```bash
python evaluation/evaluator.py \
    --output eval_$(date +%Y%m%d).json
```

### 4. Read Summaries

Always check the `_summary.txt` file for insights:

```bash
cat evaluation/eval_results_summary.txt
```

The summary includes:
- Automatic analysis of metrics
- Identification of issues
- Performance assessment
- Actionable insights

---

## Troubleshooting

**Q: Test set not found?**
- Run `python evaluation/test_set_builder.py` first
- Check `data/processed/test_set.jsonl` exists

**Q: Retrieval metrics all 0.0?**
- Check Qdrant is running: `docker ps`
- Check `vtk_docs` collection exists
- Verify chunk IDs match between test set and index

**Q: End-to-end evaluation fails?**
- Check `.env` has valid API key
- Check LLM provider is configured correctly
- Try `--num-examples 1` for debugging
- Check rate limits

---

## Dependencies

**None!** Pure Python standard library:
- `json` - Data loading
- `re` - Pattern matching  
- `difflib` - Code similarity
- `dataclasses` - Structured data
- `math` - Metric calculations

The evaluation module has zero external dependencies.
