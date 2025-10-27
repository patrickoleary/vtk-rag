#!/usr/bin/env python3
"""
Visual Regression Tests for Generated VTK Code

Tests execute generated code in Docker sandbox and compare
rendered output against baseline images.

Usage:
    # Run with visual testing enabled
    RUN_VISUAL_TESTS=1 python tests/visual_testing/test_visual_regression.py
    
    # Update baselines
    UPDATE_BASELINES=1 python tests/visual_testing/test_visual_regression.py
"""

import os
import unittest
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from docker_sandbox import DockerSandbox
from visual_regression import VisualRegressionTester


# Check if visual testing is enabled
RUN_VISUAL_TESTS = os.getenv('RUN_VISUAL_TESTS', '0') == '1'
UPDATE_BASELINES = os.getenv('UPDATE_BASELINES', '0') == '1'


class TestVTKValidation(unittest.TestCase):
    """Integration tests for VTK code validation - validates generated code produces correct output"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests"""
        # Skip all tests if visual testing not enabled
        if not RUN_VISUAL_TESTS:
            raise unittest.SkipTest("Visual tests disabled. Set RUN_VISUAL_TESTS=1 to enable")
        
        # Create Docker sandbox
        cls.sandbox = DockerSandbox(
            image_name="vtk-visual-testing",
            memory_limit="512m",
            timeout=30
        )
        
        # Check if image exists, build if needed
        if not cls.sandbox.image_exists():
            dockerfile_dir = Path(__file__).parent
            print(f"\nBuilding Docker image from {dockerfile_dir}...")
            success = cls.sandbox.build_image(dockerfile_dir)
            if not success:
                raise RuntimeError("Failed to build Docker image")
        
        # Create visual tester - use baselines from tests directory
        # Integration test baselines should be with other test data
        baseline_dir = Path(__file__).parent.parent / "tests" / "visual_testing" / "baselines"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        cls.tester = VisualRegressionTester(
            baseline_dir=baseline_dir,
            threshold=0.95  # 95% similarity required
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if hasattr(cls, 'sandbox'):
            cls.sandbox.cleanup()
    
    def test_simple_cylinder(self):
        """Test: Simple cylinder rendering"""
        
        code = """
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow

# Create cylinder
cylinder = vtkCylinderSource()
cylinder.SetResolution(50)

# Create mapper
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())

# Create actor
actor = vtkActor()
actor.SetMapper(mapper)

# Create renderer
renderer = vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.4)

# Create render window
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)
render_window.OffScreenRenderingOn()
render_window.Render()
"""
        
        # Execute code in sandbox
        result = self.sandbox.execute_code(code, "simple_cylinder")
        
        # Skip if VTK rendering failed (segfault - needs GPU/custom build)
        if not result['success']:
            error_msg = str(result.get('error', ''))
            if '139' in error_msg or 'Segmentation' in error_msg:
                self.skipTest("VTK offscreen rendering not available (segfault - requires GPU or custom VTK build)")
            else:
                self.fail(f"Code execution failed: {result.get('error')}")
        
        # Skip if no image captured
        if result['output_image'] is None:
            self.skipTest("No image captured - VTK rendering may require GPU")
        
        # Update baseline if requested
        if UPDATE_BASELINES:
            self.tester.update_baseline(result['output_image'], "simple_cylinder")
            self.skipTest("Baseline updated")
        
        # Compare with baseline
        comparison = self.tester.compare_with_baseline(
            result['output_image'],
            "simple_cylinder"
        )
        
        # Report results
        print(f"\n{comparison['message']}")
        if comparison['has_baseline']:
            print(f"  Similarity: {comparison['similarity']:.3f}")
            print(f"  Pixel diff: {comparison['pixel_diff_percent']:.2f}%")
            if comparison['diff_image_path']:
                print(f"  Diff image: {comparison['diff_image_path']}")
        
        # Assert test passed
        self.assertTrue(comparison['has_baseline'], "No baseline - run with UPDATE_BASELINES=1")
        self.assertTrue(comparison['passed'], f"Visual regression detected: {comparison['message']}")
    
    def test_colored_sphere(self):
        """Test: Colored sphere rendering"""
        
        code = """
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow

# Create sphere
sphere = vtkSphereSource()
sphere.SetThetaResolution(30)
sphere.SetPhiResolution(30)

# Create mapper
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(sphere.GetOutputPort())

# Create actor with color
actor = vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1.0, 0.3, 0.3)  # Red

# Create renderer
renderer = vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.9, 0.9, 0.9)

# Create render window
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)
render_window.OffScreenRenderingOn()
render_window.Render()
"""
        
        result = self.sandbox.execute_code(code, "colored_sphere")
        
        # Skip if VTK rendering failed (segfault - needs GPU/custom build)
        if not result['success']:
            error_msg = str(result.get('error', ''))
            if '139' in error_msg or 'Segmentation' in error_msg:
                self.skipTest("VTK offscreen rendering not available (segfault - requires GPU or custom VTK build)")
            else:
                self.fail(f"Code execution failed: {result.get('error')}")
        
        # Skip if no image captured
        if result['output_image'] is None:
            self.skipTest("No image captured - VTK rendering may require GPU")
        
        if UPDATE_BASELINES:
            self.tester.update_baseline(result['output_image'], "colored_sphere")
            self.skipTest("Baseline updated")
        
        comparison = self.tester.compare_with_baseline(
            result['output_image'],
            "colored_sphere"
        )
        
        print(f"\n{comparison['message']}")
        if comparison['has_baseline']:
            print(f"  Similarity: {comparison['similarity']:.3f}")
        
        self.assertTrue(comparison['has_baseline'], "No baseline - run with UPDATE_BASELINES=1")
        self.assertTrue(comparison['passed'], f"Visual regression detected: {comparison['message']}")
    
    def test_cone_with_rotation(self):
        """Test: Cone with rotation"""
        
        code = """
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkFiltersSources import vtkConeSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow

# Create cone
cone = vtkConeSource()
cone.SetResolution(50)

# Create mapper
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cone.GetOutputPort())

# Create actor with rotation
actor = vtkActor()
actor.SetMapper(mapper)
actor.RotateX(30)
actor.RotateY(45)

# Create renderer
renderer = vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.2, 0.3, 0.4)

# Create render window
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)
render_window.OffScreenRenderingOn()
render_window.Render()
"""
        
        result = self.sandbox.execute_code(code, "cone_with_rotation")
        
        # Skip if VTK rendering failed (segfault - needs GPU/custom build)
        if not result['success']:
            error_msg = str(result.get('error', ''))
            if '139' in error_msg or 'Segmentation' in error_msg:
                self.skipTest("VTK offscreen rendering not available (segfault - requires GPU or custom VTK build)")
            else:
                self.fail(f"Code execution failed: {result.get('error')}")
        
        # Skip if no image captured
        if result['output_image'] is None:
            self.skipTest("No image captured - VTK rendering may require GPU")
        
        if UPDATE_BASELINES:
            self.tester.update_baseline(result['output_image'], "cone_with_rotation")
            self.skipTest("Baseline updated")
        
        comparison = self.tester.compare_with_baseline(
            result['output_image'],
            "cone_with_rotation"
        )
        
        print(f"\n{comparison['message']}")
        if comparison['has_baseline']:
            print(f"  Similarity: {comparison['similarity']:.3f}")
        
        self.assertTrue(comparison['has_baseline'], "No baseline - run with UPDATE_BASELINES=1")
        self.assertTrue(comparison['passed'], f"Visual regression detected: {comparison['message']}")




if __name__ == '__main__':
    unittest.main()
