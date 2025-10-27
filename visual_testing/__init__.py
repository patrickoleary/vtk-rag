"""
Visual Regression Testing for VTK Generated Code

Provides secure Docker-based execution and visual comparison of VTK code output.
"""

from .docker_sandbox import DockerSandbox
from .visual_regression import VisualRegressionTester

__all__ = ['DockerSandbox', 'VisualRegressionTester']
