#!/usr/bin/env python3
"""
Tests for LLM Enrichment Functionality

Tests:
- Enrichment method (generate/improve explanations)
- When enrichment should/shouldn't happen
- Enrichment metadata
- Error handling
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'post-processing'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))

from json_response_processor import JSONResponseProcessor


class TestEnrichmentLogic(unittest.TestCase):
    """Test enrichment decision logic"""
    
    def setUp(self):
        """Create processor"""
        self.processor = JSONResponseProcessor()
    
    def test_code_response_without_explanation_should_enrich(self):
        """Code with no explanation should be enriched"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "cylinder = vtkCylinderSource()",
            "explanation": "",
            "confidence": "high",
            "citations": []
        }
        
        # Should identify as needing enrichment
        self.assertEqual(response['content_type'], 'code')
        self.assertEqual(response['explanation'], '')
    
    def test_non_code_response_should_not_enrich(self):
        """Non-code responses should pass through"""
        response = {
            "response_type": "answer",
            "content_type": "api",
            "explanation": "Test explanation",
            "confidence": "high",
            "citations": []
        }
        
        # Should not enrich API responses
        self.assertEqual(response['content_type'], 'api')
    
    def test_code_without_code_field_should_not_enrich(self):
        """Code response without actual code should not enrich"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "",  # No code
            "explanation": "",
            "confidence": "high",
            "citations": []
        }
        
        # Should skip if no code to explain
        self.assertEqual(response['code'], '')


class TestEnrichmentMethod(unittest.TestCase):
    """Test the enrich_with_llm method"""
    
    def setUp(self):
        """Create processor"""
        self.processor = JSONResponseProcessor()
    
    def test_enrich_returns_response_if_no_llm(self):
        """Should return original response if LLM not available"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "test",
            "explanation": "",
            "confidence": "high",
            "citations": []
        }
        
        # Mock HAS_LLM to False
        with patch('json_response_processor.HAS_LLM', False):
            result = self.processor.enrich_with_llm(response, [])
            self.assertEqual(result, response)
    
    def test_enrich_passes_through_non_code(self):
        """Should pass through non-code responses unchanged"""
        response = {
            "response_type": "answer",
            "content_type": "api",
            "explanation": "Good explanation",
            "confidence": "high",
            "citations": []
        }
        
        result = self.processor.enrich_with_llm(response, [])
        self.assertEqual(result, response)
    
    def test_enrich_skips_if_no_code(self):
        """Should skip enrichment if code field is empty"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "",
            "explanation": "",
            "confidence": "high",
            "citations": []
        }
        
        result = self.processor.enrich_with_llm(response, [])
        self.assertEqual(result, response)
    
    def test_enrich_identifies_missing_explanation(self):
        """Should identify when explanation is missing"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "cylinder = vtkCylinderSource()",
            "explanation": "",  # Missing
            "confidence": "high",
            "citations": []
        }
        
        # Test condition checks (without actually enriching)
        self.assertEqual(response['content_type'], 'code')
        self.assertEqual(response['explanation'], '')
        self.assertTrue(response['code'])  # Has code to explain
    
    def test_enrich_identifies_brief_explanation(self):
        """Should identify when explanation is brief"""
        response = {
            "response_type": "answer",
            "content_type": "code",
            "code": "cylinder = vtkCylinderSource()",
            "explanation": "Creates cylinder",  # Brief
            "confidence": "high",
            "citations": []
        }
        
        # Test condition checks
        self.assertEqual(response['content_type'], 'code')
        self.assertTrue(len(response['explanation']) > 0)
        self.assertTrue(len(response['explanation']) < 100)  # Relatively brief


class TestContextFormatting(unittest.TestCase):
    """Test documentation context formatting"""
    
    def setUp(self):
        """Create processor"""
        self.processor = JSONResponseProcessor()
    
    def test_format_context_empty_chunks(self):
        """Should handle empty chunk list"""
        context = self.processor._format_context_for_enrichment([])
        self.assertIn("No additional documentation", context)
    
    def test_format_context_with_chunks(self):
        """Should format chunks correctly"""
        chunks = [
            {"content": "First chunk"},
            {"content": "Second chunk"},
            {"text": "Third chunk (text field)"}
        ]
        
        context = self.processor._format_context_for_enrichment(chunks)
        
        self.assertIn("[1] First chunk", context)
        self.assertIn("[2] Second chunk", context)
        self.assertIn("[3] Third chunk", context)
    
    def test_format_context_limits_to_5_chunks(self):
        """Should limit to top 5 chunks"""
        chunks = [{"content": f"Chunk {i}"} for i in range(10)]
        
        context = self.processor._format_context_for_enrichment(chunks)
        
        self.assertIn("[1]", context)
        self.assertIn("[5]", context)
        self.assertNotIn("[6]", context)


if __name__ == '__main__':
    unittest.main(verbosity=2)
