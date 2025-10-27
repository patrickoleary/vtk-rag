#!/usr/bin/env python3
"""
Test Validation Integration

Tests the complete validation workflow using unittest framework.
"""

import unittest
import sys
from pathlib import Path

# Add api-mcp directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api-mcp"))

from vtk_validator import load_validator


class TestValidationIntegration(unittest.TestCase):
    """Test VTK code validation integration"""
    
    @classmethod
    def setUpClass(cls):
        """Load validator once for all tests"""
        cls.validator = load_validator()
    
    def test_01_validator_loaded(self):
        """Test that validator loaded successfully"""
        self.assertIsNotNone(self.validator)
        self.assertGreater(len(self.validator.api.classes), 2900, "Should have loaded VTK API")
    
    def test_02_validate_correct_code(self):
        """Test validation of correct code"""
        correct_code = """
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer
from vtkmodules.vtkCommonDataModel import vtkPolyData

def main():
    mapper = vtkPolyDataMapper()
    actor = vtkActor()
    actor.SetMapper(mapper)
    renderer = vtkRenderer()
    renderer.AddActor(actor)
"""
        result = self.validator.validate_code(correct_code)
        
        self.assertTrue(result.is_valid, "Correct code should validate")
        self.assertEqual(len(result.errors), 0, "Should have no errors")
    
    def test_03_detect_wrong_import_module(self):
        """Test detection of wrong import module"""
        wrong_import = """
from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper  # Wrong module!

mapper = vtkPolyDataMapper()
"""
        result = self.validator.validate_code(wrong_import)
        
        self.assertFalse(result.is_valid, "Should detect wrong import")
        self.assertGreater(len(result.errors), 0, "Should have errors")
        
        # Check error is about import
        error_messages = [e.message for e in result.errors]
        has_import_error = any('module' in msg.lower() or 'import' in msg.lower() 
                              for msg in error_messages)
        self.assertTrue(has_import_error, "Should have import-related error")
    
    def test_04_detect_misspelled_class(self):
        """Test detection of misspelled class name"""
        misspelled_class = """
from vtkmodules.vtkRenderingCore import vtkActor

renderer = vtkRendererr()  # Misspelled!
renderer.AddActor(actor)
"""
        result = self.validator.validate_code(misspelled_class)
        
        self.assertFalse(result.is_valid, "Should detect misspelled class")
        self.assertGreater(len(result.errors), 0, "Should have errors")
    
    def test_05_detect_multiple_errors(self):
        """Test detection of multiple errors in one code block"""
        multiple_errors = """
from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper  # Wrong module
from vtkmodules.vtkRenderingCore import vtkActor

def main():
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    actor = vtkActor()
    actor.SetMapper(mapper)
    
    renderer = vtkRendererr()  # Misspelled
    renderer.AddActor(actor)
"""
        result = self.validator.validate_code(multiple_errors)
        
        self.assertFalse(result.is_valid, "Should detect multiple errors")
        self.assertGreater(len(result.errors), 1, "Should have multiple errors")
    
    def test_06_error_formatting(self):
        """Test error message formatting"""
        error_code = """
from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper

mapper = vtkPolyDataMapper()
"""
        result = self.validator.validate_code(error_code)
        
        if result.has_errors:
            formatted = result.format_errors()
            self.assertIsInstance(formatted, str)
            self.assertGreater(len(formatted), 0, "Should have formatted error message")
    
    def test_07_import_validation_correct(self):
        """Test import validation with correct import"""
        result = self.validator.api.validate_import(
            "from vtkmodules.vtkRenderingCore import vtkPolyDataMapper"
        )
        
        self.assertTrue(result['valid'], "Should validate correct import")
    
    def test_08_import_validation_wrong(self):
        """Test import validation with wrong module"""
        result = self.validator.api.validate_import(
            "from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper"
        )
        
        self.assertFalse(result['valid'], "Should detect wrong module")
        self.assertIn('suggested', result)
    
    def test_09_import_validation_multi_class(self):
        """Test import validation with multiple classes"""
        result = self.validator.api.validate_import(
            "from vtkmodules.vtkRenderingCore import vtkActor, vtkRenderer"
        )
        
        self.assertTrue(result['valid'], "Should validate multi-class import")
    
    def test_10_empty_code(self):
        """Test validation of empty code"""
        result = self.validator.validate_code("")
        
        # Empty code should be valid (no errors to detect)
        self.assertTrue(result.is_valid)
    
    def test_11_code_with_only_comments(self):
        """Test validation of code with only comments"""
        comment_code = """
# This is a comment
# Another comment
"""
        result = self.validator.validate_code(comment_code)
        
        # Comments-only code should be valid
        self.assertTrue(result.is_valid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
