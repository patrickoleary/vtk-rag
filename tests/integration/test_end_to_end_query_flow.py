#!/usr/bin/env python3
"""
Integration Tests for End-to-End Query Flow

Tests the complete flow from query → classification → routing → handler → JSON output
Uses mocked LLM and retriever to test integration without external dependencies.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'retrieval-pipeline'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'grounding-prompting'))

from sequential_pipeline import SequentialPipeline
from prompt_templates import VTKPromptTemplate


class TestEndToEndAPIQuery(unittest.TestCase):
    """Test complete API query flow"""
    
    def setUp(self):
        """Set up mocked pipeline"""
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
        
        self.template = VTKPromptTemplate()
    
    def test_api_query_complete_flow(self):
        """Test API query from start to finish"""
        # Mock retriever
        mock_chunk = Mock()
        mock_chunk.content = "vtkActor.SetMapper() documentation"
        mock_chunk.chunk_id = "api_001"
        self.mock_retriever.retrieve_api_doc.return_value = [mock_chunk]
        
        # Mock LLM response
        api_response = {
            "response_type": "answer",
            "content_type": "api",
            "explanation": "SetMapper assigns a mapper to an actor",
            "parameters": [{"name": "mapper", "type": "vtkMapper", "description": "The mapper"}],
            "confidence": "high",
            "citations": [{"number": 1, "reason": "API documentation"}]
        }
        self.mock_llm_client.generate_json.return_value = api_response
        
        # Execute
        result = self.pipeline.process_query("What does SetMapper do?")
        
        # Verify
        self.assertEqual(result['response_type'], 'answer')
        self.assertEqual(result['content_type'], 'api')
        self.assertIn('explanation', result)
        self.assertIn('parameters', result)
        
        # Verify retriever was called
        self.mock_retriever.retrieve_api_doc.assert_called_once()
        
        # Verify LLM was called with correct schema
        self.mock_llm_client.generate_json.assert_called_once()
        call_args = self.mock_llm_client.generate_json.call_args
        self.assertEqual(call_args[1]['schema_name'], 'APILookupOutput')


class TestEndToEndDataQuery(unittest.TestCase):
    """Test complete data→code query flow"""
    
    def setUp(self):
        """Set up mocked pipeline"""
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    def test_data_query_complete_flow(self):
        """Test data→code query with alternatives"""
        # Mock retriever
        mock_chunk1 = Mock()
        mock_chunk1.content = "Point cloud example"
        mock_chunk1.metadata = {
            'has_data_files': True,
            'data_files': [{'filename': 'points.csv'}],
            'category': 'Point Cloud'
        }
        
        mock_chunk2 = Mock()
        mock_chunk2.content = "Scatter plot example"
        mock_chunk2.metadata = {
            'has_data_files': True,
            'data_files': [{'filename': 'data.csv'}],
            'category': '3D Scatter'
        }
        
        self.mock_retriever.retrieve_code.return_value = [mock_chunk1, mock_chunk2]
        
        # Mock LLM response
        data_response = {
            "response_type": "answer",
            "content_type": "code",
            "data_analysis": "CSV with 3D coordinates",
            "suggested_techniques": ["Point cloud", "3D scatter"],
            "code": "# Point cloud code",
            "explanation": "Creates point cloud",
            "alternative_approaches": [
                {
                    "technique": "3D Scatter",
                    "description": "Use vtkGlyph3D",
                    "vtk_classes": ["vtkGlyph3D"],
                    "complexity": "moderate"
                }
            ],
            "confidence": "high",
            "citations": [{"number": 1, "reason": "CSV example"}]
        }
        self.mock_llm_client.generate_json.return_value = data_response
        
        # Execute
        result = self.pipeline.process_query("I have points.csv, what can I do?")
        
        # Verify
        self.assertEqual(result['content_type'], 'code')
        self.assertIn('suggested_techniques', result)
        self.assertIn('alternative_approaches', result)
        self.assertEqual(len(result['alternative_approaches']), 1)
        
        # Verify correct schema was used
        call_args = self.mock_llm_client.generate_json.call_args
        self.assertEqual(call_args[1]['schema_name'], 'DataToCodeOutput')


class TestEndToEndCodeToDataQuery(unittest.TestCase):
    """Test complete code→data query flow"""
    
    def setUp(self):
        """Set up mocked pipeline"""
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        # Set up mock client attribute for the search method
        self.mock_retriever.client = Mock()
        self.mock_retriever.collection_name = "vtk_docs"
        self.mock_retriever.embedding_model = Mock()
        # Mock encode to return an object with tolist() method
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2]
        self.mock_retriever.embedding_model.encode.return_value = mock_embedding
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    def test_code_to_data_complete_flow(self):
        """Test code→data query with file extraction"""
        # Code to analyze
        code = """
from vtkmodules.vtkIOGeometry import vtkSTLReader
reader = vtkSTLReader()
reader.SetFileName('mesh.stl')
"""
        
        # Mock search results
        mock_result = Mock()
        mock_result.metadata = {
            'title': 'STL Example',
            'data_files': [{'filename': 'mesh.stl', 'description': 'Test mesh'}],
            'data_download_info': [{'url': 'https://example.com/mesh.stl', 'size': '2MB'}]
        }
        self.mock_retriever.client.search.return_value = [Mock(payload=mock_result.metadata)]
        self.mock_retriever._format_results.return_value = [mock_result]
        
        # Mock LLM response
        data_response = {
            "response_type": "answer",
            "content_type": "data",
            "explanation": "Code reads STL files",
            "code_requirements": "Expects STL mesh",
            "data_files": [
                {
                    "filename": "mesh.stl",
                    "download_url": "https://example.com/mesh.stl",
                    "file_type": "STL",
                    "file_size": "2MB"
                }
            ],
            "confidence": "high",
            "citations": [{"number": 1, "reason": "STL example"}]
        }
        self.mock_llm_client.generate_json.return_value = data_response
        
        # Execute
        result = self.pipeline.process_query(
            "Do you have example data for this?",
            code=code
        )
        
        # Verify
        self.assertEqual(result['content_type'], 'data')
        self.assertIn('data_files', result)
        self.assertEqual(len(result['data_files']), 1)
        self.assertEqual(result['data_files'][0]['filename'], 'mesh.stl')
        
        # Verify correct schema
        call_args = self.mock_llm_client.generate_json.call_args
        self.assertEqual(call_args[1]['schema_name'], 'CodeToDataOutput')


class TestPromptTemplateIntegration(unittest.TestCase):
    """Test prompt templates integrate properly with handlers"""
    
    def setUp(self):
        self.template = VTKPromptTemplate()
    
    def test_all_prompt_methods_return_valid_structure(self):
        """All prompt methods should return valid instruction dicts"""
        methods = [
            'get_api_lookup_instructions',
            'get_explanation_instructions',
            'get_data_to_code_instructions',
            'get_code_to_data_instructions',
        ]
        
        for method_name in methods:
            with self.subTest(method=method_name):
                method = getattr(self.template, method_name)
                instructions = method()
                
                # Verify structure
                self.assertIsInstance(instructions, dict)
                self.assertIn('role', instructions)
                self.assertIn('task', instructions)
                self.assertIn('output_format_example', instructions)
                self.assertIn('requirements', instructions)
                self.assertIn('grounding', instructions)
                
                # Verify output format has required fields
                example = instructions['output_format_example']
                self.assertIn('response_type', example)
                self.assertIn('confidence', example)
                self.assertIn('citations', example)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that old code still works"""
    
    def setUp(self):
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    @patch.object(SequentialPipeline, '_handle_code_query')
    def test_old_generate_method_works(self, mock_handler):
        """Legacy generate() method should still work"""
        mock_handler.return_value = {
            'query': 'test',
            'code': 'print("test")',
            'explanation': 'Test'
        }
        
        # Old way should still work
        result = self.pipeline.generate("Create a cylinder")
        
        # Should return PipelineResult
        self.assertEqual(result.query, 'test')
        self.assertEqual(result.code, 'print("test")')


if __name__ == '__main__':
    unittest.main(verbosity=2)
