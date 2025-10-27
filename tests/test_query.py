#!/usr/bin/env python3
"""
Tests for unified query.py system

Tests the complete query flow including:
- Code generation queries
- API documentation queries
- Concept explanation queries
- Visual testing integration
- Output file generation
"""

import unittest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module under test
import query


class TestPrerequisites(unittest.TestCase):
    """Test prerequisite checking"""
    
    def test_check_prerequisites_no_visual(self):
        """Test basic prerequisite check without visual testing"""
        issues = query.check_prerequisites(require_visual=False)
        # Should return a list (empty if all OK, or with issues)
        self.assertIsInstance(issues, list)
    
    @patch('requests.get')
    def test_qdrant_unavailable(self, mock_get):
        """Test detection when Qdrant is not running"""
        mock_get.side_effect = Exception("Connection failed")
        
        issues = query.check_prerequisites(require_visual=False)
        
        # Should detect Qdrant issue
        qdrant_issues = [i for i in issues if 'Qdrant' in i]
        self.assertTrue(len(qdrant_issues) > 0)
    
    @patch('subprocess.run')
    def test_docker_unavailable_with_visual_flag(self, mock_run):
        """Test detection when Docker is unavailable but visual testing requested"""
        mock_run.side_effect = Exception("Docker not found")
        
        issues = query.check_prerequisites(require_visual=True)
        
        # Should detect Docker issue
        docker_issues = [i for i in issues if 'Docker' in i]
        self.assertTrue(len(docker_issues) > 0)


class TestQueryVTK(unittest.TestCase):
    """Test main query_vtk function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_query = "How do I create a cylinder in VTK?"
        self.temp_dir = tempfile.mkdtemp()
    
    @patch('query.SequentialPipeline')
    def test_basic_code_query(self, mock_pipeline_class):
        """Test basic code generation query"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Mock response
        mock_pipeline.process_query.return_value = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'import vtk\nprint("Hello")',
            'explanation': 'This creates a cylinder',
            'citations': ['CylinderExample_0']
        }
        
        # Run query
        response = query.query_vtk(
            self.test_query,
            visual_test=False,
            enrich=False,
            verbose=False
        )
        
        # Verify
        self.assertEqual(response['content_type'], 'code')
        self.assertIn('code', response)
        self.assertIn('explanation', response)
        self.assertIn('citations', response)
    
    @patch('query.SequentialPipeline')
    def test_api_documentation_query(self, mock_pipeline_class):
        """Test API documentation lookup"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_pipeline.process_query.return_value = {
            'response_type': 'direct',
            'content_type': 'api',
            'class_name': 'vtkPolyDataMapper',
            'methods': ['SetInputData', 'SetInputConnection', 'Update'],
            'explanation': 'Mapper for polygonal data',
            'citations': ['vtkPolyDataMapper_api']
        }
        
        response = query.query_vtk(
            "What methods does vtkPolyDataMapper have?",
            verbose=False
        )
        
        self.assertEqual(response['content_type'], 'api')
        self.assertIn('class_name', response)
        self.assertIn('methods', response)
    
    @patch('query.SequentialPipeline')
    def test_output_file_saved(self, mock_pipeline_class):
        """Test that output is saved to file"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_response = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'test code',
            'explanation': 'test explanation'
        }
        mock_pipeline.process_query.return_value = mock_response
        
        output_file = Path(self.temp_dir) / 'test_output.json'
        
        response = query.query_vtk(
            self.test_query,
            output_file=str(output_file),
            verbose=False
        )
        
        # Verify file was created
        self.assertTrue(output_file.exists())
        
        # Verify content
        with open(output_file) as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['code'], 'test code')
    
    @patch('query.SequentialPipeline')
    @patch('query.JSONResponseProcessor')
    def test_enrichment_flag(self, mock_processor_class, mock_pipeline_class):
        """Test that enrichment is applied when flag is set"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.last_retrieved_chunks = []
        
        mock_response = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'test',
            'explanation': 'original'
        }
        mock_pipeline.process_query.return_value = mock_response
        
        # Mock processor
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        enriched_response = mock_response.copy()
        enriched_response['_enrichment'] = {'was_enriched': True}
        mock_processor.enrich_with_llm.return_value = enriched_response
        
        response = query.query_vtk(
            self.test_query,
            enrich=True,
            verbose=False
        )
        
        # Verify enrichment was called
        mock_processor.enrich_with_llm.assert_called_once()
        self.assertIn('_enrichment', response)
    
    @patch('query.SequentialPipeline')
    @patch('visual_evaluator.VisualEvaluator')
    def test_visual_testing_success(self, mock_evaluator_class, mock_pipeline_class):
        """Test visual testing with successful execution"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_response = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'import vtk\nprint("test")',
            'explanation': 'test'
        }
        mock_pipeline.process_query.return_value = mock_response
        
        # Mock visual evaluator
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        mock_exec_result = Mock()
        mock_exec_result.success = True
        mock_exec_result.execution_time = 1.5
        mock_exec_result.has_visual_output = True
        mock_exec_result.error = None
        mock_evaluator.execute_code.return_value = mock_exec_result
        
        response = query.query_vtk(
            self.test_query,
            visual_test=True,
            verbose=False
        )
        
        # Verify visual validation was added
        self.assertIn('_visual_validation', response)
        self.assertTrue(response['_visual_validation']['execution_success'])
        self.assertEqual(response['_visual_validation']['execution_time'], 1.5)
    
    @patch('query.SequentialPipeline')
    @patch('visual_evaluator.VisualEvaluator')
    def test_visual_testing_failure(self, mock_evaluator_class, mock_pipeline_class):
        """Test visual testing with execution failure"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_response = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'invalid code',
            'explanation': 'test'
        }
        mock_pipeline.process_query.return_value = mock_response
        
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        mock_exec_result = Mock()
        mock_exec_result.success = False
        mock_exec_result.execution_time = 0
        mock_exec_result.has_visual_output = False
        mock_exec_result.error = "SyntaxError: invalid syntax"
        mock_evaluator.execute_code.return_value = mock_exec_result
        
        response = query.query_vtk(
            self.test_query,
            visual_test=True,
            verbose=False
        )
        
        # Verify failure is captured
        self.assertIn('_visual_validation', response)
        self.assertFalse(response['_visual_validation']['execution_success'])
        self.assertIn('error', response['_visual_validation'])


class TestFormatResponse(unittest.TestCase):
    """Test response formatting"""
    
    def test_format_code_response(self):
        """Test formatting of code response"""
        response = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'import vtk\nprint("test")',
            'explanation': 'This is a test',
            'citations': ['example1', 'example2']
        }
        
        formatted = query.format_response(response)
        
        self.assertIn('CODE:', formatted)
        self.assertIn('```python', formatted)
        self.assertIn('EXPLANATION:', formatted)
        self.assertIn('CITATIONS:', formatted)
    
    def test_format_api_response(self):
        """Test formatting of API documentation response"""
        response = {
            'response_type': 'direct',
            'content_type': 'api',
            'class_name': 'vtkActor',
            'methods': ['SetMapper', 'GetMapper', 'SetProperty'],
            'explanation': 'Represents an actor in the scene'
        }
        
        formatted = query.format_response(response)
        
        self.assertIn('CLASS: vtkActor', formatted)
        self.assertIn('Methods:', formatted)
        self.assertIn('SetMapper', formatted)
    
    def test_format_with_data_files(self):
        """Test formatting when data files are present"""
        response = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'test',
            'explanation': 'test',
            'data_files': [
                {'filename': 'test.vtp', 'url': 'http://example.com/test.vtp'},
                {'filename': 'data.stl', 'url': 'http://example.com/data.stl'}
            ]
        }
        
        formatted = query.format_response(response)
        
        self.assertIn('DATA FILES:', formatted)
        self.assertIn('test.vtp', formatted)
        self.assertIn('data.stl', formatted)
    
    def test_format_with_image_url(self):
        """Test formatting when image URL is present"""
        response = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'test',
            'explanation': 'test',
            'image_url': 'http://example.com/image.png'
        }
        
        formatted = query.format_response(response)
        
        self.assertIn('IMAGE:', formatted)
        self.assertIn('http://example.com/image.png', formatted)
    
    def test_format_with_visual_validation(self):
        """Test formatting when visual validation results are present"""
        response = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'test',
            'explanation': 'test',
            '_visual_validation': {
                'execution_success': True,
                'execution_time': 2.5,
                'has_visual_output': True,
                'error': None
            }
        }
        
        formatted = query.format_response(response)
        
        self.assertIn('VISUAL VALIDATION:', formatted)
        self.assertIn('SUCCESS', formatted)
        self.assertIn('2.5', formatted)


class TestMainFunction(unittest.TestCase):
    """Test CLI argument parsing and main function"""
    
    @patch('query.query_vtk')
    @patch('query.check_prerequisites')
    def test_basic_cli_call(self, mock_check, mock_query):
        """Test basic CLI call"""
        mock_check.return_value = []  # No issues
        mock_query.return_value = {
            'response_type': 'direct',
            'content_type': 'code',
            'code': 'test',
            'explanation': 'test'
        }
        
        with patch('sys.argv', ['query.py', 'How do I create a cylinder?']):
            result = query.main()
        
        self.assertEqual(result, 0)
        mock_query.assert_called_once()
    
    @patch('query.check_prerequisites')
    def test_prerequisites_failure(self, mock_check):
        """Test that main exits when prerequisites fail"""
        mock_check.return_value = ['Qdrant not running', 'Docker not available']
        
        with patch('sys.argv', ['query.py', 'test query']):
            result = query.main()
        
        self.assertEqual(result, 1)
    
    @patch('query.query_vtk')
    @patch('query.check_prerequisites')
    def test_visual_test_flag(self, mock_check, mock_query):
        """Test that --visual-test flag is passed through"""
        mock_check.return_value = []
        mock_query.return_value = {'response_type': 'direct', 'content_type': 'code'}
        
        with patch('sys.argv', ['query.py', '--visual-test', 'test query']):
            query.main()
        
        # Verify visual_test=True was passed
        call_kwargs = mock_query.call_args[1]
        self.assertTrue(call_kwargs['visual_test'])
    
    @patch('query.query_vtk')
    @patch('query.check_prerequisites')
    def test_enrich_flag(self, mock_check, mock_query):
        """Test that --enrich flag is passed through"""
        mock_check.return_value = []
        mock_query.return_value = {'response_type': 'direct', 'content_type': 'code'}
        
        with patch('sys.argv', ['query.py', '--enrich', 'test query']):
            query.main()
        
        call_kwargs = mock_query.call_args[1]
        self.assertTrue(call_kwargs['enrich'])
    
    @patch('query.query_vtk')
    @patch('query.check_prerequisites')
    def test_output_file_flag(self, mock_check, mock_query):
        """Test that --output flag is passed through"""
        mock_check.return_value = []
        mock_query.return_value = {'response_type': 'direct', 'content_type': 'code'}
        
        with patch('sys.argv', ['query.py', '--output', 'result.json', 'test query']):
            query.main()
        
        call_kwargs = mock_query.call_args[1]
        self.assertEqual(call_kwargs['output_file'], 'result.json')
    
    @patch('query.query_vtk')
    @patch('query.check_prerequisites')
    def test_quiet_flag(self, mock_check, mock_query):
        """Test that --quiet flag suppresses verbose output"""
        mock_check.return_value = []
        mock_query.return_value = {'response_type': 'direct', 'content_type': 'code'}
        
        with patch('sys.argv', ['query.py', '--quiet', 'test query']):
            query.main()
        
        call_kwargs = mock_query.call_args[1]
        self.assertFalse(call_kwargs['verbose'])


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
