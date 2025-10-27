#!/usr/bin/env python3
"""
Visual Code Evaluator

Executes generated VTK code in Docker sandbox and evaluates visual output.
Integrates with the E2E evaluation pipeline to provide execution and visual metrics.

Key Metrics:
1. Execution Success - Does code run without errors?
2. Visual Output - Does code produce rendered output?
3. Visual Regression - Does output differ from baseline?

Usage:
    from visual_evaluator import VisualEvaluator
    
    evaluator = VisualEvaluator(
        enable_execution=True,
        baseline_dir=Path("tests/visual_testing/baselines"),
        create_baselines=False
    )
    
    metrics = evaluator.evaluate_code(
        code="...",
        test_name="query_123",
        expected_has_output=True
    )
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'visual_testing'))

try:
    from docker_sandbox import DockerSandbox
    from visual_regression import VisualRegressionTester
    VISUAL_TESTING_AVAILABLE = True
except ImportError:
    VISUAL_TESTING_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualEvaluator:
    """
    Evaluates generated VTK code by executing it and comparing visual output.
    
    Provides three key metrics:
    1. execution_success - Code runs without errors
    2. has_visual_output - Code produces rendered output
    3. visual_regression_detected - Output differs from baseline
    """
    
    def __init__(
        self,
        enable_execution: bool = True,
        baseline_dir: Optional[Path] = None,
        create_baselines: bool = False,
        memory_limit: str = "512m",
        timeout: int = 30,
        similarity_threshold: float = 0.95
    ):
        """
        Initialize visual evaluator.
        
        Args:
            enable_execution: Whether to execute code (requires Docker)
            baseline_dir: Directory for baseline images
            create_baselines: Automatically create missing baselines
            memory_limit: Docker memory limit (e.g., "512m")
            timeout: Execution timeout in seconds
            similarity_threshold: SSIM threshold for visual comparison (0-1)
        """
        self.enable_execution = enable_execution
        self.create_baselines = create_baselines
        self.similarity_threshold = similarity_threshold
        
        if not VISUAL_TESTING_AVAILABLE:
            logger.warning("Visual testing modules not available - execution disabled")
            self.enable_execution = False
        
        if self.enable_execution:
            # Initialize Docker sandbox
            self.sandbox = DockerSandbox(
                image_name="vtk-visual-testing",
                memory_limit=memory_limit,
                timeout=timeout
            )
            
            # Check if Docker image exists
            if not self.sandbox.image_exists():
                logger.error("Docker image 'vtk-visual-testing' not found")
                logger.error("Build it with: cd visual_testing && docker build -t vtk-visual-testing .")
                self.enable_execution = False
            
            # Initialize visual regression tester
            if baseline_dir is None:
                baseline_dir = Path(__file__).parent.parent / "tests" / "visual_testing" / "baselines"
            
            baseline_dir.mkdir(parents=True, exist_ok=True)
            
            self.tester = VisualRegressionTester(
                baseline_dir=baseline_dir,
                threshold=similarity_threshold
            )
            
            logger.info(f"Visual evaluator initialized (baseline_dir={baseline_dir})")
        else:
            self.sandbox = None
            self.tester = None
            logger.info("Visual evaluator initialized (execution disabled)")
    
    def evaluate_code(
        self,
        code: str,
        test_name: str,
        expected_has_output: bool = True
    ) -> Dict[str, Any]:
        """
        Execute code and return visual validation metrics.
        
        Args:
            code: Generated VTK code to execute
            test_name: Unique identifier for this test
            expected_has_output: Whether code should produce visual output
        
        Returns:
            Dictionary with metrics:
            {
                # Execution metrics
                'execution_attempted': bool,
                'execution_success': bool,
                'execution_time': float,
                'execution_error': str | None,
                
                # Visual output metrics
                'has_visual_output': bool,
                'visual_output_size': int,
                
                # Visual comparison metrics
                'has_baseline': bool,
                'visual_similarity': float | None,
                'visual_regression_detected': bool | None,
                'visual_diff_percent': float | None,
                'baseline_created': bool,
                
                # Overall result
                'visual_validation_passed': bool
            }
        """
        # Default metrics (execution disabled)
        if not self.enable_execution:
            return {
                'execution_attempted': False,
                'execution_success': None,
                'execution_time': 0,
                'execution_error': "Visual testing disabled",
                'has_visual_output': None,
                'visual_output_size': 0,
                'has_baseline': False,
                'visual_similarity': None,
                'visual_regression_detected': None,
                'visual_diff_percent': None,
                'baseline_created': False,
                'visual_validation_passed': None
            }
        
        # Execute code in sandbox
        logger.debug(f"Executing code for test '{test_name}'")
        result = self.sandbox.execute_code(code, test_name)
        
        metrics = {
            'execution_attempted': True,
            'execution_success': result['success'],
            'execution_time': result.get('execution_time', 0),
            'execution_error': result.get('error'),
            'baseline_created': False
        }
        
        # Check for visual output
        if result['success'] and result.get('output_image'):
            metrics['has_visual_output'] = True
            metrics['visual_output_size'] = len(result['output_image'])
            
            logger.debug(f"Visual output captured: {metrics['visual_output_size']} bytes")
            
            # Create baseline if requested and missing
            if self.create_baselines and not self.tester.baseline_exists(test_name):
                logger.info(f"Creating baseline for '{test_name}'")
                self.tester.update_baseline(result['output_image'], test_name)
                metrics['baseline_created'] = True
            
            # Compare with baseline
            comparison = self.tester.compare_with_baseline(
                result['output_image'],
                test_name
            )
            
            metrics['has_baseline'] = comparison['has_baseline']
            
            if comparison['has_baseline']:
                metrics['visual_similarity'] = comparison.get('similarity')
                metrics['visual_regression_detected'] = not comparison['passed']
                metrics['visual_diff_percent'] = comparison.get('diff_percent')
                
                if comparison['passed']:
                    logger.debug(f"Visual comparison passed (similarity={comparison['similarity']:.3f})")
                else:
                    logger.warning(f"Visual regression detected (similarity={comparison['similarity']:.3f})")
            else:
                metrics['visual_similarity'] = None
                metrics['visual_regression_detected'] = None
                metrics['visual_diff_percent'] = None
                logger.debug(f"No baseline for '{test_name}' - skipping comparison")
        else:
            # No visual output
            metrics['has_visual_output'] = False
            metrics['visual_output_size'] = 0
            metrics['has_baseline'] = False
            metrics['visual_similarity'] = None
            metrics['visual_regression_detected'] = None
            metrics['visual_diff_percent'] = None
            
            if expected_has_output:
                logger.warning(f"Expected visual output but none captured for '{test_name}'")
        
        # Compute overall pass/fail
        # Pass if:
        # 1. Code executed successfully AND
        # 2. (No output expected OR output was captured) AND
        # 3. (No baseline OR no regression detected)
        metrics['visual_validation_passed'] = (
            metrics['execution_success'] and
            (not expected_has_output or metrics['has_visual_output']) and
            (not metrics['has_baseline'] or not metrics['visual_regression_detected'])
        )
        
        return metrics
    
    def cleanup(self):
        """Clean up resources (close Docker sandbox)"""
        if self.sandbox:
            self.sandbox.cleanup()
            logger.debug("Visual evaluator cleaned up")


# Example usage
if __name__ == '__main__':
    import os
    
    # Enable visual testing
    os.environ['RUN_VISUAL_TESTS'] = '1'
    
    # Initialize evaluator
    evaluator = VisualEvaluator(
        enable_execution=True,
        create_baselines=False
    )
    
    # Test code
    test_code = """
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow

cylinder = vtkCylinderSource()
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())
actor = vtkActor()
actor.SetMapper(mapper)
renderer = vtkRenderer()
renderer.AddActor(actor)
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)
render_window.Render()
"""
    
    # Evaluate
    result = evaluator.evaluate_code(
        code=test_code,
        test_name="example_cylinder",
        expected_has_output=True
    )
    
    # Print results
    print("\nVisual Evaluation Results:")
    print(f"  Execution Success: {result['execution_success']}")
    print(f"  Has Visual Output: {result['has_visual_output']}")
    print(f"  Visual Similarity: {result['visual_similarity']}")
    print(f"  Overall Passed: {result['visual_validation_passed']}")
    
    evaluator.cleanup()
