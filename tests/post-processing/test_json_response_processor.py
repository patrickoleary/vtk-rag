#!/usr/bin/env python3
"""
Unit Tests for JSON Response Processor

Tests JSON response validation, enrichment, and metadata extraction.
"""

import unittest
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'post-processing'))

from json_response_processor import JSONResponseProcessor, EnrichedResponse, process_response


class TestJSONResponseProcessor(unittest.TestCase):
    """Test JSON response processing"""
    
    def setUp(self):
        """Create processor"""
        self.processor = JSONResponseProcessor()
    
    def test_process_code_response(self):
        """Test processing a code response"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "from vtkmodules.vtkFiltersSources import vtkCylinderSource\ncylinder = vtkCylinderSource()",
            "explanation": "Creates a cylinder using vtkCylinderSource",
            "vtk_classes_used": ["vtkCylinderSource"],
            "citations": [{"number": 1, "reason": "cylinder example"}],
            "confidence": "high"
        }
        
        enriched = self.processor.process(response)
        
        self.assertIsInstance(enriched, EnrichedResponse)
        self.assertEqual(enriched.response_type, "answer")
        self.assertEqual(enriched.content_type, "code")
        self.assertTrue(enriched.has_code)
        self.assertTrue(enriched.has_citations)
        self.assertEqual(enriched.citation_count, 1)
        self.assertEqual(enriched.confidence, "high")
        self.assertIn("vtkCylinderSource", enriched.vtk_classes)
    
    def test_process_api_response(self):
        """Test processing an API response"""
        response = {
            "response_type": "answer",
            "content_type": "api",
            "explanation": "SetMapper assigns a vtkMapper to a vtkActor",
            "parameters": [{"name": "mapper", "type": "vtkMapper", "description": "The mapper"}],
            "citations": [],
            "confidence": "high"
        }
        
        enriched = self.processor.process(response)
        
        self.assertEqual(enriched.content_type, "api")
        self.assertFalse(enriched.has_code)
        self.assertFalse(enriched.has_citations)
        self.assertIn("vtkMapper", enriched.vtk_classes)
        self.assertIn("vtkActor", enriched.vtk_classes)
    
    def test_process_data_response(self):
        """Test processing a data response"""
        response = {
            "response_type": "answer",
            "content_type": "data",
            "explanation": "STL files available",
            "data_files": [
                {"filename": "mesh.stl", "download_url": "http://example.com/mesh.stl"},
                {"filename": "sphere.stl", "download_url": "http://example.com/sphere.stl"}
            ],
            "citations": [{"number": 1, "reason": "STL examples"}],
            "confidence": "high"
        }
        
        enriched = self.processor.process(response)
        
        self.assertEqual(enriched.content_type, "data")
        self.assertTrue(enriched.metadata['has_data_files'])
        self.assertEqual(enriched.metadata['data_file_count'], 2)
        self.assertTrue(enriched.metadata['has_download_urls'])
    
    def test_extract_vtk_classes_from_code(self):
        """Test VTK class extraction from code"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "cylinder = vtkCylinderSource()\nmapper = vtkPolyDataMapper()\nactor = vtkActor()",
            "confidence": "high",
            "citations": []
        }
        
        classes = self.processor._extract_vtk_classes(response)
        
        self.assertIn("vtkCylinderSource", classes)
        self.assertIn("vtkPolyDataMapper", classes)
        self.assertIn("vtkActor", classes)
        self.assertEqual(len(classes), 3)
    
    def test_extract_vtk_classes_from_explanation(self):
        """Test VTK class extraction from explanation"""
        response = {
            "response_type": "answer",
            "content_type": "explanation",
            "explanation": "The vtkActor uses a vtkMapper to render data",
            "confidence": "high",
            "citations": []
        }
        
        classes = self.processor._extract_vtk_classes(response)
        
        self.assertIn("vtkActor", classes)
        self.assertIn("vtkMapper", classes)
    
    def test_validate_response_valid(self):
        """Test validation of valid response"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "test",
            "confidence": "high",
            "citations": []
        }
        
        # Should not raise
        self.processor._validate_response(response)
    
    def test_validate_response_missing_type(self):
        """Test validation catches missing response_type"""
        response = {
            "content_type": "code",
            "code": "test"
        }
        
        with self.assertRaises(ValueError) as cm:
            self.processor._validate_response(response)
        
        self.assertIn("response_type", str(cm.exception))
    
    def test_validate_response_invalid_type(self):
        """Test validation catches invalid response_type"""
        response = {
            "response_type": "invalid_type",
            "content_type": "code"
        }
        
        with self.assertRaises(ValueError) as cm:
            self.processor._validate_response(response)
        
        self.assertIn("Invalid response_type", str(cm.exception))
    
    def test_validate_response_missing_content_type(self):
        """Test validation catches missing content_type for answers"""
        response = {
            "response_type": "answer",
            # Missing content_type
        }
        
        with self.assertRaises(ValueError) as cm:
            self.processor._validate_response(response)
        
        self.assertIn("content_type", str(cm.exception))
    
    def test_validate_citations_valid(self):
        """Test citation validation with valid citations"""
        response = {
            "citations": [
                {"number": 1, "reason": "First source"},
                {"number": 2, "reason": "Second source"}
            ]
        }
        
        result = self.processor.validate_citations(response)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['citation_count'], 2)
        self.assertEqual(result['citation_numbers'], [1, 2])
    
    def test_validate_citations_missing_field(self):
        """Test citation validation catches missing fields"""
        response = {
            "citations": [
                {"number": 1},  # Missing reason
                {"reason": "test"}  # Missing number
            ]
        }
        
        result = self.processor.validate_citations(response)
        
        self.assertFalse(result['valid'])
        self.assertTrue(len(result['issues']) > 0)
    
    def test_validate_citations_duplicate_numbers(self):
        """Test citation validation catches duplicates"""
        response = {
            "citations": [
                {"number": 1, "reason": "First"},
                {"number": 1, "reason": "Duplicate"}
            ]
        }
        
        result = self.processor.validate_citations(response)
        
        self.assertFalse(result['valid'])
        self.assertTrue(any("Duplicate" in issue for issue in result['issues']))
    
    def test_extract_mentioned_files(self):
        """Test file extraction from response"""
        response = {
            "code": "reader.SetFileName('data.csv')\nreader2.SetFileName('mesh.stl')",
            "data_files": [
                {"filename": "points.csv"},
                {"filename": "surface.stl"}
            ]
        }
        
        files = self.processor.extract_mentioned_files(response)
        
        self.assertIn("points.csv", files['data_files'])
        self.assertIn("surface.stl", files['data_files'])
        self.assertIn("data.csv", files['code_files'])
        self.assertIn("mesh.stl", files['code_files'])
    
    def test_summarize(self):
        """Test response summarization"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "cylinder = vtkCylinderSource()\nmapper = vtkPolyDataMapper()",
            "explanation": "Creates cylinder",
            "vtk_classes_used": ["vtkCylinderSource", "vtkPolyDataMapper"],
            "citations": [{"number": 1, "reason": "example"}],
            "confidence": "high"
        }
        
        summary = self.processor.summarize(response)
        
        self.assertIn("answer", summary)
        self.assertIn("code", summary)
        self.assertIn("high", summary)
        self.assertIn("vtkCylinderSource", summary)
        self.assertIn("Citations", summary)
    
    def test_convenience_function(self):
        """Test convenience function"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "test",
            "confidence": "high",
            "citations": []
        }
        
        enriched = process_response(response)
        
        self.assertIsInstance(enriched, EnrichedResponse)
        self.assertEqual(enriched.response_type, "answer")


class TestMetadataExtraction(unittest.TestCase):
    """Test metadata extraction features"""
    
    def setUp(self):
        """Create processor"""
        self.processor = JSONResponseProcessor()
    
    def test_metadata_for_code_response(self):
        """Test metadata extraction for code response"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "import vtk\ncylinder = vtkCylinderSource()",
            "confidence": "high",
            "citations": []
        }
        
        enriched = self.processor.process(response)
        
        self.assertIn('code_length', enriched.metadata)
        self.assertIn('has_imports', enriched.metadata)
        self.assertTrue(enriched.metadata['has_imports'])
    
    def test_metadata_for_data_response_with_alternatives(self):
        """Test metadata for data response with alternatives"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "test",
            "alternative_approaches": [
                {"technique": "A", "description": "desc"},
                {"technique": "B", "description": "desc"}
            ],
            "confidence": "high",
            "citations": []
        }
        
        enriched = self.processor.process(response)
        
        self.assertTrue(enriched.metadata['has_alternatives'])
    
    def test_vtk_class_deduplication(self):
        """Test VTK class deduplication"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "actor1 = vtkActor()\nactor2 = vtkActor()\nmapper = vtkMapper()",
            "explanation": "Uses vtkActor twice",
            "confidence": "high",
            "citations": []
        }
        
        enriched = self.processor.process(response)
        
        # vtkActor appears multiple times but extraction deduplicates
        self.assertEqual(len(enriched.vtk_classes), 2)  # vtkActor, vtkMapper (deduplicated)
        self.assertEqual(enriched.metadata['vtk_class_count'], 2)  # After dedup
        self.assertEqual(enriched.metadata['unique_vtk_classes'], 2)  # Same as count


if __name__ == '__main__':
    unittest.main(verbosity=2)
