# Visual Regression Testing for Generated VTK Code

Executes generated code in isolated Docker containers and compares rendered output against baseline images.

## Features

- 🐳 **Docker Isolation** - Secure execution in containers
- 🖼️ **Visual Comparison** - SSIM-based image comparison
- 📊 **Diff Generation** - Automatic visual diff images
- 🔒 **Resource Limits** - Memory, CPU, and timeout constraints
- 🎯 **Baseline Management** - Create and update reference images

---

## Quick Start

### 1. Build Docker Image

```bash
cd visual_testing
docker build -t vtk-visual-testing .
```

Or use the build script:
```bash
python build_docker_image.py
```

### 2. Create Baselines

First run creates baseline images:

```bash
UPDATE_BASELINES=1 RUN_VISUAL_TESTS=1 python test_vtk_validation.py -v
```

### 3. Run Visual Tests

Compare against baselines:

```bash
RUN_VISUAL_TESTS=1 python test_vtk_validation.py -v
```

---

## Usage

### Running Tests

**Normal test run (compare with baselines):**
```bash
RUN_VISUAL_TESTS=1 python test_vtk_validation.py
```

**Update baselines:**
```bash
UPDATE_BASELINES=1 RUN_VISUAL_TESTS=1 python test_vtk_validation.py
```

**Run specific test:**
```bash
RUN_VISUAL_TESTS=1 python test_vtk_validation.py TestVTKValidation.test_simple_cylinder
```

### Using Docker Sandbox

```python
from docker_sandbox import DockerSandbox

# Initialize sandbox
sandbox = DockerSandbox(
    memory_limit="512m",
    cpu_quota=50000,  # 50% CPU
    timeout=30  # seconds
)

# Execute VTK code
# IMPORTANT: Must include vtkRenderingOpenGL2 for offscreen rendering
code = """
import vtkmodules.vtkRenderingOpenGL2  # Required for OpenGL backend
from vtkmodules.vtkFiltersSources import vtkCylinderSource
# ... VTK code ...
"""

result = sandbox.execute_code(code, "my_test")

if result['success']:
    print(f"✓ Executed in {result['execution_time']:.2f}s")
    if result['output_image']:
        print(f"✓ Captured {len(result['output_image'])} bytes")
else:
    print(f"✗ Failed: {result['error']}")
```

### Using Visual Regression Tester

```python
from visual_regression import VisualRegressionTester
from pathlib import Path

# Initialize tester
tester = VisualRegressionTester(
    baseline_dir=Path("tests/visual_testing/baselines"),
    threshold=0.95  # 95% similarity required
)

# Compare with baseline
comparison = tester.compare_with_baseline(
    test_image=image_bytes,
    test_name="my_test"
)

if comparison['passed']:
    print(f"✓ Visual test passed (similarity={comparison['similarity']:.3f})")
else:
    print(f"✗ Regression detected")
    print(f"  Diff image: {comparison['diff_image_path']}")

# Update baseline
tester.update_baseline(image_bytes, "my_test")
```

---

## Architecture

### Components

```
visual_testing/
├── Dockerfile              # Docker image with VTK 9.5.2 + OSMesa
├── requirements.txt        # Python dependencies for Docker
├── docker_sandbox.py       # Executes code in Docker containers
├── visual_regression.py    # Compares images, manages baselines
├── test_vtk_validation.py  # Integration test suite (unittest)
└── README.md               # This file

tests/visual_testing/
├── baselines/              # Baseline images (test data)
│   ├── simple_cylinder.png
│   ├── colored_sphere.png
│   └── cone_with_rotation.png
└── test_docker_sandbox.py  # Unit tests for Docker sandbox
```

### Workflow

```
1. Generate VTK code
   ↓
2. Execute in Docker sandbox
   - Isolated environment
   - Resource limits
   - Offscreen rendering
   ↓
3. Capture rendered image
   - Save as PNG
   - Extract from render window
   ↓
4. Compare with baseline
   - SSIM similarity
   - Pixel difference
   - Generate diff image
   ↓
5. Pass/Fail decision
   - similarity >= threshold → Pass
   - similarity < threshold → Fail
```

---

## Configuration

### Docker Sandbox

```python
DockerSandbox(
    image_name="vtk-visual-testing",
    memory_limit="512m",      # Memory limit per container
    cpu_quota=50000,          # 50% of one CPU (100000 = 1 CPU)
    timeout=30,               # Execution timeout in seconds
    network_disabled=True     # Disable network access
)
```

### Visual Regression Tester

```python
VisualRegressionTester(
    baseline_dir=Path("baselines"),
    threshold=0.95            # SSIM threshold (0-1)
)
```

### Threshold Guidelines

| Threshold | Meaning | Use Case |
|-----------|---------|----------|
| 0.99 | Nearly identical | Exact pixel-perfect matching |
| 0.95 | Very similar | Default (allows minor rendering differences) |
| 0.90 | Similar | Tolerates some visual changes |
| 0.85 | Loosely similar | Very tolerant |

---

## Docker Image

### What's Included

- **Python 3.13** - Base environment
- **VTK 9.5.2** - VTK Python bindings with OSMesa support
- **Xvfb** - Virtual framebuffer for offscreen rendering
- **Mesa** - Software OpenGL implementation
- **Image libraries** - OpenCV, Pillow, scikit-image

### Environment

- **User:** Non-root `vtkuser` (UID 1000)
- **Workspace:** `/workspace` (read-only)
- **Output:** `/output` (write)
- **Display:** `:99` (Xvfb)

### Building

```bash
# Build image
docker build -t vtk-visual-testing .

# Test image
docker run --rm vtk-visual-testing python3 -c "import vtk; print(vtk.vtkVersion.GetVTKVersion())"
```

---

## Testing Generated Code

### Integration with Pipeline

```python
from sequential_pipeline import SequentialPipeline
from docker_sandbox import DockerSandbox
from visual_regression import VisualRegressionTester

# Generate code
pipeline = SequentialPipeline()
result = pipeline.process_query("Create a cylinder")

# Security validation (automatic)
if not result['security_check_passed']:
    print("⚠️ Security issues:", result['security_issues'])
    exit(1)

# Execute in sandbox
sandbox = DockerSandbox()
exec_result = sandbox.execute_code(
    result['code'],
    "generated_cylinder"
)

# Visual validation (optional)
if exec_result['success'] and exec_result['output_image']:
    tester = VisualRegressionTester(Path("baselines"))
    comparison = tester.compare_with_baseline(
        exec_result['output_image'],
        "generated_cylinder"
    )
    
    if comparison['passed']:
        print("✓ Generated code renders correctly")
    else:
        print("✗ Visual regression detected")
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Visual Regression Tests

on: [push, pull_request]

jobs:
  visual-tests:
    runs-on: ubuntu-latest
    
    services:
      docker:
        image: docker:20.10
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest docker pillow scikit-image opencv-python-headless
      
      - name: Build Docker image
        run: |
          cd visual_testing
          docker build -t vtk-visual-testing .
      
      - name: Run visual tests
        run: |
          RUN_VISUAL_TESTS=1 bash run_all_tests.sh
      
      - name: Upload diff images on failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: visual-diffs
          path: tests/visual_testing/baselines/*_diff.png
```

---

## Troubleshooting

### Docker Image Won't Build

**Error:** `Cannot connect to Docker daemon`
```bash
# Start Docker service
sudo systemctl start docker

# Or use Docker Desktop (Mac/Windows)
```

**Error:** `Permission denied`
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

### Tests Failing

**No baseline images:**
```bash
# Create baselines first
UPDATE_BASELINES=1 RUN_VISUAL_TESTS=1 python test_vtk_validation.py
```

**Visual regression detected:**
- Check diff images in `baselines/*_diff.png`
- Left: baseline, Middle: test, Right: difference (enhanced)
- Update baseline if change is expected: `UPDATE_BASELINES=1 ...`

**Timeout errors:**
- Increase timeout: `DockerSandbox(timeout=60)`
- Check if code has infinite loops
- Review Docker container logs

### Performance Issues

**Slow execution:**
- Increase CPU quota: `cpu_quota=100000` (1 full CPU)
- Increase memory: `memory_limit="1g"`
- Use local Docker (not remote)

**Large images:**
- Reduce render window size in code
- Use lower resolution
- Compress baselines (PNG already compressed)

---

## Best Practices

### ✅ DO

- Create baselines from known-good output
- Run visual tests in CI/CD pipeline
- Update baselines when rendering changes are expected
- Review diff images before updating baselines
- Use reasonable SSIM thresholds (0.95 is good default)

### ❌ DON'T

- Commit diff images to git (they're temporary)
- Set threshold too high (0.99+) - minor GPU differences will fail
- Set threshold too low (<0.90) - won't catch real regressions
- Run without baselines (will fail)
- Skip security validation before visual testing

---

## Metrics

### What's Measured

| Metric | Description | Range |
|--------|-------------|-------|
| **SSIM** | Structural similarity | 0-1 (1 = identical) |
| **Pixel Diff Count** | Number of different pixels | 0-N |
| **Pixel Diff %** | Percentage of different pixels | 0-100% |
| **Execution Time** | Code execution duration | seconds |

### Interpreting Results

**SSIM Scores:**
- 1.000 = Identical
- 0.99+ = Nearly identical (minor GPU differences)
- 0.95-0.99 = Very similar (rendering variations)
- 0.90-0.95 = Similar (noticeable differences)
- <0.90 = Different (likely regression)

---

## Examples

See `test_vtk_validation.py` for integration test examples:
- Simple cylinder rendering (with OpenGL2 import)
- Colored sphere
- Rotated cone

See `tests/visual_testing/test_docker_sandbox.py` for unit test examples:
- Resource limit tests
- Timeout tests
- Isolation tests
- Network isolation tests

---

## Dependencies

**Host requirements:**
- Docker
- Python 3.8+

**Python packages:**
```
docker>=6.0.0
Pillow>=10.0.0
scikit-image>=0.21.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
```

Install:
```bash
pip install docker Pillow scikit-image opencv-python-headless numpy
```

---

## Summary

**Phase 2 Visual Testing provides:**

✅ **Secure execution** - Docker isolation with resource limits  
✅ **Visual validation** - Compare rendered output with baselines  
✅ **Regression detection** - Catch unintended visual changes  
✅ **CI/CD ready** - Automated testing in pipelines  
✅ **Developer friendly** - Easy baseline management  

**Use for:**
- Testing generated VTK code
- Catching visual regressions
- Validating rendering changes
- CI/CD quality gates
