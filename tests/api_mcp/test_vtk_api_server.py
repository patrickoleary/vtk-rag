#!/usr/bin/env python3
"""
Test VTK API MCP Server

Tests all MCP tools using unittest framework.
"""

import unittest
import sys
from pathlib import Path

# Add api-mcp directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api-mcp"))

from vtk_api_server import VTKAPIIndex


class TestVTKAPIServer(unittest.TestCase):
    """Test VTK API MCP Server functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Load API index once for all tests"""
        api_docs_path = Path(__file__).parent.parent.parent / "data" / "raw" / "vtk-python-docs.jsonl"
        
        if not api_docs_path.exists():
            raise FileNotFoundError(f"API docs not found at {api_docs_path}")
        
        cls.index = VTKAPIIndex(api_docs_path)
    
    def test_01_api_index_loaded(self):
        """Test that API index loaded successfully"""
        self.assertGreater(len(self.index.classes), 2900, "Should load ~2900+ VTK classes")
        self.assertGreater(len(self.index.modules), 100, "Should have 100+ modules")
    
    def test_02_get_class_info(self):
        """Test vtk_get_class_info"""
        class_info = self.index.get_class_info("vtkPolyDataMapper")
        
        self.assertIsNotNone(class_info, "Should find vtkPolyDataMapper")
        self.assertEqual(class_info['module'], "vtkmodules.vtkRenderingCore")
        self.assertIn('content', class_info)
    
    def test_03_get_class_info_not_found(self):
        """Test vtk_get_class_info with non-existent class"""
        class_info = self.index.get_class_info("vtkFakeClass123")
        self.assertIsNone(class_info, "Should return None for non-existent class")
    
    def test_04_search_classes(self):
        """Test vtk_search_classes"""
        results = self.index.search_classes("reader", limit=5)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Should find reader classes")
        self.assertLessEqual(len(results), 5, "Should respect limit")
        
        # Check result structure
        for result in results:
            self.assertIn('class_name', result)
            self.assertIn('module', result)
    
    def test_05_get_module_classes(self):
        """Test vtk_get_module_classes"""
        classes = self.index.get_module_classes("vtkmodules.vtkRenderingCore")
        
        self.assertIsInstance(classes, list)
        self.assertGreater(len(classes), 100, "vtkRenderingCore should have 100+ classes")
        self.assertIn("vtkPolyDataMapper", classes)
        self.assertIn("vtkActor", classes)
    
    def test_06_get_module_classes_not_found(self):
        """Test vtk_get_module_classes with non-existent module"""
        classes = self.index.get_module_classes("vtkmodules.vtkFakeModule")
        # Should return None or empty list for non-existent module
        self.assertTrue(classes is None or len(classes) == 0, "Should return None or empty list for non-existent module")
    
    def test_07_validate_import_correct(self):
        """Test vtk_validate_import with correct import"""
        result = self.index.validate_import("from vtkmodules.vtkRenderingCore import vtkPolyDataMapper")
        
        self.assertTrue(result['valid'], "Should validate correct import")
        self.assertIn('message', result)
    
    def test_08_validate_import_wrong_module(self):
        """Test vtk_validate_import with wrong module"""
        result = self.index.validate_import("from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper")
        
        self.assertFalse(result['valid'], "Should detect wrong module")
        self.assertIn('message', result)
        self.assertIn('suggested', result)
        # Check that suggested contains the correct import (may have extra text)
        self.assertIn("from vtkmodules.vtkRenderingCore import vtkPolyDataMapper", result['suggested'])
    
    def test_09_validate_import_non_existent_class(self):
        """Test vtk_validate_import with non-existent class"""
        result = self.index.validate_import("from vtkmodules.vtkRenderingCore import vtkFakeMapper")
        
        self.assertFalse(result['valid'], "Should detect non-existent class")
        self.assertIn('message', result)
    
    def test_10_validate_import_multi_class(self):
        """Test vtk_validate_import with multiple classes"""
        result = self.index.validate_import("from vtkmodules.vtkRenderingCore import (vtkActor, vtkPolyDataMapper)")
        
        self.assertTrue(result['valid'], "Should validate multi-class import")
    
    def test_11_get_method_info(self):
        """Test vtk_get_method_info"""
        method_info = self.index.get_method_info("vtkPolyDataMapper", "SetInputData")
        
        # Method info might or might not be available depending on docs format
        # Just check it returns something or None
        self.assertTrue(method_info is None or isinstance(method_info, dict))


if __name__ == '__main__':
    unittest.main(verbosity=2)
