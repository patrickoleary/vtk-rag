#!/usr/bin/env python3
"""Tests for LLM Client JSON Generation"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))

from llm_client import LLMClient, LLMProvider


class TestLLMClientJSONGeneration(unittest.TestCase):
    """Test JSON generation functionality"""
    
    def setUp(self):
        """Set up mock LLM client"""
        # Mock the client initialization
        with patch.dict('os.environ', {
            'LLM_PROVIDER': 'anthropic',
            'ANTHROPIC_API_KEY': 'test-key',
            'LLM_MODEL': 'test-model'
        }):
            self.client = LLMClient()
            # Mock the actual API client
            self.client.client = Mock()
    
    def test_extract_json_from_clean_response(self):
        """Test extracting JSON from clean response"""
        response = '{"key": "value", "number": 42}'
        result = self.client._extract_json_from_response(response)
        
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)
    
    def test_extract_json_from_markdown_block(self):
        """Test extracting JSON from markdown code block"""
        response = '''Here's the JSON:
```json
{"key": "value", "nested": {"data": true}}
```
That's the result.'''
        
        result = self.client._extract_json_from_response(response)
        self.assertEqual(result["key"], "value")
        self.assertTrue(result["nested"]["data"])
    
    def test_extract_json_from_markdown_without_language(self):
        """Test extracting JSON from markdown without json tag"""
        response = '''```
{"key": "value"}
```'''
        
        result = self.client._extract_json_from_response(response)
        self.assertEqual(result["key"], "value")
    
    def test_extract_json_with_text_before_and_after(self):
        """Test extracting JSON when there's text around it"""
        response = '''Some text before
{"key": "value", "array": [1, 2, 3]}
Some text after'''
        
        result = self.client._extract_json_from_response(response)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["array"], [1, 2, 3])
    
    def test_extract_json_invalid_raises_error(self):
        """Test that invalid JSON raises error"""
        response = "This is not JSON at all"
        
        with self.assertRaises(json.JSONDecodeError):
            self.client._extract_json_from_response(response)
    
    def test_validate_decomposition_output_valid(self):
        """Test validating valid decomposition output"""
        data = {
            "understanding": "test",
            "requires_visualization": True,
            "libraries_needed": ["vtk"],
            "data_files": [],
            "steps": []
        }
        
        result = self.client._validate_json_schema(data, "DecompositionOutput")
        self.assertTrue(result)
    
    def test_validate_decomposition_output_invalid(self):
        """Test validating invalid decomposition output"""
        data = {
            "understanding": "test"
            # Missing required fields
        }
        
        result = self.client._validate_json_schema(data, "DecompositionOutput")
        self.assertFalse(result)
    
    def test_validate_generation_output_valid(self):
        """Test validating valid generation output"""
        data = {
            "step_number": 1,
            "understanding": "test",
            "imports": [],
            "code": "test",
            "citations": []
        }
        
        result = self.client._validate_json_schema(data, "GenerationOutput")
        self.assertTrue(result)
    
    def test_validate_unknown_schema_returns_true(self):
        """Test that unknown schema name returns True (no validation)"""
        data = {"any": "data"}
        result = self.client._validate_json_schema(data, "UnknownSchema")
        self.assertTrue(result)
    
    def test_get_schema_details_decomposition(self):
        """Test getting schema details for DecompositionOutput"""
        details = self.client._get_schema_details("DecompositionOutput")
        
        # Should contain required field names
        self.assertIn("understanding", details)
        self.assertIn("requires_visualization", details)
        self.assertIn("libraries_needed", details)
        self.assertIn("data_files", details)
        self.assertIn("steps", details)
        self.assertIn("step_number", details)
        self.assertIn("description", details)
        self.assertIn("search_query", details)
        self.assertIn("focus", details)
    
    def test_get_schema_details_generation(self):
        """Test getting schema details for GenerationOutput"""
        details = self.client._get_schema_details("GenerationOutput")
        
        self.assertIn("step_number", details)
        self.assertIn("understanding", details)
        self.assertIn("imports", details)
        self.assertIn("code", details)
        self.assertIn("citations", details)
    
    def test_get_schema_details_validation(self):
        """Test getting schema details for ValidationOutput"""
        details = self.client._get_schema_details("ValidationOutput")
        
        self.assertIn("fixed_code", details)
        self.assertIn("changes_made", details)
        self.assertIn("error_type", details)
        self.assertIn("fix", details)
    
    @patch.object(LLMClient, 'generate')
    def test_generate_json_success_first_try(self, mock_generate):
        """Test successful JSON generation on first try"""
        mock_generate.return_value = '{"understanding": "test", "requires_visualization": true, "libraries_needed": ["vtk"], "data_files": [], "steps": []}'
        
        result = self.client.generate_json(
            prompt_data={"query": "test"},
            schema_name="DecompositionOutput"
        )
        
        self.assertEqual(result["understanding"], "test")
        self.assertEqual(mock_generate.call_count, 1)
    
    @patch.object(LLMClient, 'generate')
    def test_generate_json_retry_on_invalid_json(self, mock_generate):
        """Test retry when LLM returns invalid JSON"""
        # First call returns invalid JSON, second returns valid
        mock_generate.side_effect = [
            "This is not JSON",
            '{"understanding": "test", "requires_visualization": true, "libraries_needed": [], "data_files": [], "steps": []}'
        ]
        
        result = self.client.generate_json(
            prompt_data={"query": "test"},
            schema_name="DecompositionOutput",
            max_retries=2
        )
        
        self.assertEqual(result["understanding"], "test")
        self.assertEqual(mock_generate.call_count, 2)
    
    @patch.object(LLMClient, 'generate')
    def test_generate_json_fails_after_max_retries(self, mock_generate):
        """Test failure after max retries"""
        mock_generate.return_value = "Not JSON"
        
        with self.assertRaises(ValueError) as context:
            self.client.generate_json(
                prompt_data={"query": "test"},
                schema_name="DecompositionOutput",
                max_retries=2
            )
        
        self.assertIn("failed to return valid JSON", str(context.exception))
        self.assertEqual(mock_generate.call_count, 3)  # Initial + 2 retries
    
    @patch.object(LLMClient, 'generate')
    def test_generate_json_with_markdown_wrapping(self, mock_generate):
        """Test handling JSON wrapped in markdown"""
        mock_generate.return_value = '''```json
{
    "understanding": "test",
    "requires_visualization": false,
    "libraries_needed": ["vtk"],
    "data_files": [],
    "steps": [{"step_number": 1, "description": "test", "search_query": "test", "focus": "test"}]
}
```'''
        
        result = self.client.generate_json(
            prompt_data={"query": "test"},
            schema_name="DecompositionOutput"
        )
        
        self.assertEqual(result["understanding"], "test")
        self.assertFalse(result["requires_visualization"])


if __name__ == '__main__':
    unittest.main()
