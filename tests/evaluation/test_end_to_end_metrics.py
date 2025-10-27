"""
Unit test for end-to-end code quality metrics

Tests verify code quality assessment (exactness, correctness, syntax validity)
without requiring LLM or real pipeline execution.
"""

import sys
import unittest
from pathlib import Path

# Add evaluation module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'evaluation'))

from end_to_end_metrics import EndToEndEvaluator


class TestEndToEndMetrics(unittest.TestCase):
    """Test end-to-end code quality metrics"""
    
    def setUp(self):
        """Initialize evaluator for each test"""
        self.evaluator = EndToEndEvaluator()
    
    def test_code_quality_metrics_basic(self):
        """Test basic code quality assessment with known inputs"""
        # Gold standard code (complete VTK example)
        gold_code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper
from vtkmodules.vtkRenderingCore import vtkActor
from vtkmodules.vtkRenderingCore import vtkRenderer
from vtkmodules.vtkRenderingCore import vtkRenderWindow
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor

cylinder = vtkCylinderSource()
cylinder.SetResolution(8)

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)

renderer = vtkRenderer()
renderer.AddActor(actor)

window = vtkRenderWindow()
window.AddRenderer(renderer)

interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.Start()
"""
        
        # Generated code (similar but not identical)
        generated_code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow

cylinder = vtkCylinderSource()
cylinder.SetResolution(8)

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())

actor = vtkActor(mapper=mapper)

renderer = vtkRenderer()
renderer.AddActor(actor)

render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.Render()
"""
        
        # Calculate metrics
        exactness = self.evaluator.code_exactness(generated_code, gold_code)
        completeness = self.evaluator.code_completeness(generated_code, gold_code)
        
        # Assertions
        # Exactness: Should be < 1.0 (not identical) but > 0.0 (similar)
        self.assertGreater(exactness, 0.0)
        self.assertLess(exactness, 1.0)
        
        # Completeness: Should be high (has most components)
        # Has: imports, vtkCylinderSource, mapper, actor, renderer, window
        # Missing: vtkRenderWindowInteractor
        self.assertGreater(completeness, 0.7)
        
        # Verify it's a reasonable assessment
        print(f"\n  Code Exactness: {exactness:.3f}")
        print(f"  Code Completeness: {completeness:.3f}")
        print(f"  ✓ Metrics calculated successfully")


def run_tests():
    """Run all tests and print results"""
    print("=" * 80)
    print("Running End-to-End Metrics Tests")
    print("=" * 80)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEndToEndMetrics)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ All tests PASSED")
    else:
        print("❌ Some tests FAILED")
    print("=" * 80)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
