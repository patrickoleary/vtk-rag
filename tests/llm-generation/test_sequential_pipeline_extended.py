#!/usr/bin/env python3
"""
Unit Tests for Extended Sequential Pipeline

Tests new query classification and routing functionality:
- Query type classification
- Handler routing
- API query handler
- Explanation query handler
- Data query handler (Data→Code)
- Code-to-data query handler (Code→Data)
- Code refinement handler (Modify existing code)
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'retrieval-pipeline'))

from sequential_pipeline import SequentialPipeline


class TestQueryClassification(unittest.TestCase):
    """Test query type classification"""
    
    def setUp(self):
        """Create pipeline with mocked dependencies"""
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    def test_classify_code_query(self):
        """Should classify code generation queries"""
        queries = [
            "Create a cylinder in VTK",
            "How to render a sphere?",
            "Generate a point cloud",
        ]
        
        for query in queries:
            with self.subTest(query=query):
                result = self.pipeline._classify_query(query)
                self.assertEqual(result, "code")
    
    def test_classify_api_query(self):
        """Should classify API documentation queries"""
        queries = [
            "What does SetMapper do?",
            "Explain the GetProperty method",
            "What is vtkActor class?",
            "How does SetRadius() work?",
        ]
        
        for query in queries:
            with self.subTest(query=query):
                result = self.pipeline._classify_query(query)
                self.assertEqual(result, "api")
    
    def test_classify_explanation_query(self):
        """Should classify concept explanation queries"""
        queries = [
            "Explain the VTK pipeline workflow",
            "What is the rendering pipeline?",
            "Describe the concept of mappers",
            "What's the difference between vtkActor and vtkActor2D?",
        ]
        
        for query in queries:
            with self.subTest(query=query):
                result = self.pipeline._classify_query(query)
                self.assertEqual(result, "explanation")
    
    def test_classify_data_query(self):
        """Should classify exploratory data queries"""
        queries = [
            "I have points.csv, what can I do?",
            "What techniques can I use with mesh.stl?",
            "What are the options for data.vti?",
        ]
        
        for query in queries:
            with self.subTest(query=query):
                result = self.pipeline._classify_query(query)
                self.assertEqual(result, "data_query")
    
    def test_classify_code_to_data_query(self):
        """Should classify code-to-data queries when code is provided"""
        query = "Do you have example data for this code?"
        code = "from vtkmodules.vtkIOGeometry import vtkSTLReader"
        
        result = self.pipeline._classify_query(query, code=code)
        self.assertEqual(result, "code_to_data")
    
    def test_classify_refinement_query(self):
        """Should classify refinement queries when existing_code is provided"""
        query = "Make it blue and increase resolution"
        existing_code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource
cylinder = vtkCylinderSource()
"""
        
        result = self.pipeline._classify_query(query, existing_code=existing_code)
        self.assertEqual(result, "refinement")
    
    def test_default_to_code(self):
        """Ambiguous queries should default to code"""
        queries = [
            "cylinder",
            "visualization",
            "3D points",
        ]
        
        for query in queries:
            with self.subTest(query=query):
                result = self.pipeline._classify_query(query)
                self.assertEqual(result, "code")


class TestCodeRequirementsAnalysis(unittest.TestCase):
    """Test code analysis for reader detection"""
    
    def setUp(self):
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    def test_analyze_stl_reader(self):
        """Should detect vtkSTLReader"""
        code = """
from vtkmodules.vtkIOGeometry import vtkSTLReader
reader = vtkSTLReader()
reader.SetFileName('mesh.stl')
"""
        
        analysis = self.pipeline._analyze_code_requirements(code)
        
        self.assertEqual(analysis['reader_type'], 'vtkSTLReader')
        self.assertIn('.stl', analysis['file_extensions'])
        self.assertTrue(analysis['data_required'])
        self.assertIn('vtkSTLReader', analysis['vtk_classes'])
    
    def test_analyze_csv_reader(self):
        """Should detect pandas CSV reader"""
        code = """
import pandas as pd
df = pd.read_csv('data.csv')
"""
        
        analysis = self.pipeline._analyze_code_requirements(code)
        
        self.assertEqual(analysis['reader_type'], 'pandas_csv')
        self.assertIn('.csv', analysis['file_extensions'])
        self.assertTrue(analysis['data_required'])
    
    def test_analyze_no_reader(self):
        """Should handle code without data readers"""
        code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource
cylinder = vtkCylinderSource()
"""
        
        analysis = self.pipeline._analyze_code_requirements(code)
        
        self.assertIsNone(analysis['reader_type'])
        self.assertEqual(analysis['file_extensions'], [])
        self.assertFalse(analysis['data_required'])
        self.assertIn('vtkCylinderSource', analysis['vtk_classes'])
    
    def test_extract_vtk_classes(self):
        """Should extract all VTK classes"""
        code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

cylinder = vtkCylinderSource()
mapper = vtkPolyDataMapper()
actor = vtkActor()
"""
        
        analysis = self.pipeline._analyze_code_requirements(code)
        
        vtk_classes = analysis['vtk_classes']
        self.assertIn('vtkCylinderSource', vtk_classes)
        self.assertIn('vtkActor', vtk_classes)
        self.assertIn('vtkPolyDataMapper', vtk_classes)


class TestHandlerRouting(unittest.TestCase):
    """Test handler routing based on query type"""
    
    def setUp(self):
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    @patch.object(SequentialPipeline, '_handle_code_query')
    @patch.object(SequentialPipeline, '_classify_query')
    def test_route_to_code_handler(self, mock_classify, mock_handler):
        """Should route code queries to code handler"""
        mock_classify.return_value = "code"
        mock_handler.return_value = {"response_type": "answer"}
        
        result = self.pipeline.process_query("Create a cylinder")
        
        mock_classify.assert_called_once()
        mock_handler.assert_called_once()
        self.assertEqual(result['response_type'], 'answer')
    
    @patch.object(SequentialPipeline, '_handle_api_query')
    @patch.object(SequentialPipeline, '_classify_query')
    def test_route_to_api_handler(self, mock_classify, mock_handler):
        """Should route API queries to API handler"""
        mock_classify.return_value = "api"
        mock_handler.return_value = {"response_type": "answer", "content_type": "api"}
        
        result = self.pipeline.process_query("What does SetMapper do?")
        
        mock_classify.assert_called_once()
        mock_handler.assert_called_once()
        self.assertEqual(result['content_type'], 'api')
    
    @patch.object(SequentialPipeline, '_handle_data_query')
    @patch.object(SequentialPipeline, '_classify_query')
    def test_route_to_data_handler(self, mock_classify, mock_handler):
        """Should route data queries to data handler"""
        mock_classify.return_value = "data_query"
        mock_handler.return_value = {"content_type": "code", "suggested_techniques": []}
        
        result = self.pipeline.process_query("I have points.csv, what can I do?")
        
        mock_classify.assert_called_once()
        mock_handler.assert_called_once()
        self.assertIn('suggested_techniques', result)


class TestHelperMethods(unittest.TestCase):
    """Test helper methods"""
    
    def setUp(self):
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    def test_format_context(self):
        """Should format chunks as numbered context"""
        mock_chunk1 = Mock()
        mock_chunk1.content = "First chunk content"
        
        mock_chunk2 = Mock()
        mock_chunk2.content = "Second chunk content"
        
        chunks = [mock_chunk1, mock_chunk2]
        
        result = self.pipeline._format_context(chunks)
        
        self.assertIn("[1]", result)
        self.assertIn("[2]", result)
        self.assertIn("First chunk content", result)
        self.assertIn("Second chunk content", result)
    
    def test_extract_data_files_from_chunks(self):
        """Should extract data files with download info"""
        mock_chunk = Mock()
        mock_chunk.metadata = {
            'title': 'STL Example',
            'data_files': [{'filename': 'mesh.stl', 'description': 'Test mesh'}],
            'data_download_info': [{'url': 'https://example.com/mesh.stl', 'size': '2MB'}]
        }
        
        chunks = [mock_chunk]
        
        result = self.pipeline._extract_data_files_from_chunks(chunks)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['filename'], 'mesh.stl')
        self.assertEqual(result[0]['download_url'], 'https://example.com/mesh.stl')
        self.assertEqual(result[0]['file_type'], 'STL')
    
    def test_extract_data_files_deduplicates(self):
        """Should deduplicate data files by filename"""
        mock_chunk1 = Mock()
        mock_chunk1.metadata = {
            'data_files': [{'filename': 'mesh.stl'}],
            'data_download_info': [{'url': 'url1'}]
        }
        
        mock_chunk2 = Mock()
        mock_chunk2.metadata = {
            'data_files': [{'filename': 'mesh.stl'}],
            'data_download_info': [{'url': 'url2'}]
        }
        
        chunks = [mock_chunk1, mock_chunk2]
        
        result = self.pipeline._extract_data_files_from_chunks(chunks)
        
        # Should only have one file (deduplicated)
        self.assertEqual(len(result), 1)


class TestCodeRefinement(unittest.TestCase):
    """Test code refinement functionality"""
    
    def setUp(self):
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    def test_analyze_existing_code(self):
        """Should analyze existing code structure"""
        code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource, vtkConeSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

cylinder = vtkCylinderSource()
cylinder.SetResolution(8)

mapper = vtkPolyDataMapper()
actor = vtkActor()
"""
        
        analysis = self.pipeline._analyze_existing_code(code)
        
        # Should extract VTK classes
        self.assertIn('vtkCylinderSource', analysis['vtk_classes'])
        self.assertIn('vtkConeSource', analysis['vtk_classes'])
        self.assertIn('vtkActor', analysis['vtk_classes'])
        
        # Should extract variables
        self.assertIn('cylinder', analysis['variables'])
        self.assertEqual(analysis['variables']['cylinder'], 'vtkCylinderSource')
        
        # Should detect imports
        self.assertTrue(len(analysis['imports']) > 0)
        
        # Should detect method calls
        self.assertIn('SetResolution', analysis['method_calls'])
    
    def test_analyze_code_with_colors(self):
        """Should detect vtkNamedColors usage"""
        code = """
from vtkmodules.vtkCommonColor import vtkNamedColors
colors = vtkNamedColors()
"""
        
        analysis = self.pipeline._analyze_existing_code(code)
        self.assertTrue(analysis['has_colors'])
    
    def test_analyze_code_with_rendering(self):
        """Should detect rendering components"""
        code = """
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow
renderer = vtkRenderer()
window = vtkRenderWindow()
"""
        
        analysis = self.pipeline._analyze_existing_code(code)
        self.assertTrue(analysis['has_rendering'])
    
    @patch.object(SequentialPipeline, '_handle_refinement_query')
    @patch.object(SequentialPipeline, '_classify_query')
    def test_route_to_refinement_handler(self, mock_classify, mock_handler):
        """Should route refinement queries to refinement handler"""
        mock_classify.return_value = "refinement"
        mock_handler.return_value = {
            "response_type": "answer",
            "content_type": "code_refinement",
            "code": "modified code",
            "modifications": []
        }
        
        result = self.pipeline.process_query(
            "Make it blue",
            existing_code="some code"
        )
        
        mock_classify.assert_called_once()
        mock_handler.assert_called_once()
        self.assertEqual(result['content_type'], 'code_refinement')
    
    def test_validate_modified_code_valid(self):
        """Should validate syntactically correct code"""
        code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource
cylinder = vtkCylinderSource()
cylinder.SetResolution(50)
"""
        
        is_valid = self.pipeline._validate_modified_code(code, "original")
        self.assertTrue(is_valid)
    
    def test_validate_modified_code_invalid_syntax(self):
        """Should detect syntax errors"""
        code = """
from vtkmodules.vtkFiltersSources import vtkCylinderSource
cylinder = vtkCylinderSource(
"""
        
        is_valid = self.pipeline._validate_modified_code(code, "original")
        self.assertFalse(is_valid)
    
    def test_generate_diff(self):
        """Should generate diff between original and modified code"""
        original = """line1
line2
line3"""
        
        modified = """line1
line2_modified
line3"""
        
        diff = self.pipeline._generate_diff(original, modified, [])
        
        self.assertIn('line2', diff)
        self.assertIn('-', diff)  # Shows removed line
        self.assertIn('+', diff)  # Shows added line


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility"""
    
    def setUp(self):
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    @patch.object(SequentialPipeline, '_handle_code_query')
    def test_generate_method_still_works(self, mock_handler):
        """Legacy generate() method should still work"""
        mock_handler.return_value = {
            'query': 'test',
            'code': 'print("test")',
            'explanation': 'Test explanation'
        }
        
        # Should not raise exception
        result = self.pipeline.generate("Create a cylinder")
        
        # Should return PipelineResult
        self.assertEqual(result.query, 'test')
        self.assertEqual(result.code, 'print("test")')


if __name__ == '__main__':
    unittest.main(verbosity=2)
