"""
Tests for code_validator.py
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))

from code_validator import CodeValidator, LLMCodeValidator, ValidationError


class TestCodeValidator(unittest.TestCase):
    """Test basic code validation"""
    
    def setUp(self):
        self.validator = CodeValidator()
    
    def test_valid_code_passes(self):
        """Test that valid code passes validation"""
        code = "x = 1\nprint(x)"
        is_valid, errors = self.validator.validate_code(code)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_empty_code_fails(self):
        """Test that empty code fails validation"""
        code = ""
        is_valid, errors = self.validator.validate_code(code)
        
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "empty_code")
    
    def test_syntax_error_detected(self):
        """Test that syntax errors are detected"""
        code = "x = 1\nprint(x"  # Missing closing paren
        is_valid, errors = self.validator.validate_code(code)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertEqual(errors[0].error_type, "syntax_error")
    
    def test_no_statements_detected(self):
        """Test that code with only comments/whitespace fails"""
        code = "# Just a comment\n  \n"
        is_valid, errors = self.validator.validate_code(code)
        
        self.assertFalse(is_valid)
        self.assertEqual(errors[0].error_type, "no_statements")
    
    def test_format_errors(self):
        """Test error formatting"""
        errors = [
            ValidationError("syntax_error", "Missing paren", line=2, suggestion="Add )"),
            ValidationError("name_error", "Undefined variable", line=5)
        ]
        
        formatted = self.validator.format_errors(errors)
        
        self.assertIn("syntax_error", formatted)
        self.assertIn("line 2", formatted)
        self.assertIn("Add )", formatted)
        self.assertIn("name_error", formatted)


class TestLLMCodeValidator(unittest.TestCase):
    """Test LLM-based code validation and fixing"""
    
    def setUp(self):
        self.mock_llm = Mock()
        self.validator = LLMCodeValidator(self.mock_llm)
    
    def test_valid_code_no_fix_needed(self):
        """Test that valid code passes without calling LLM"""
        code = "x = 1\nprint(x)"
        
        fixed_code, changes, success = self.validator.validate_and_fix(
            code=code,
            context="Test",
            max_retries=2
        )
        
        self.assertEqual(fixed_code, code)
        self.assertEqual(len(changes), 0)
        self.assertTrue(success)
        self.mock_llm.generate_json.assert_not_called()
    
    def test_invalid_code_calls_llm(self):
        """Test that invalid code triggers LLM fix attempt"""
        invalid_code = "x = 1\nprint(x"  # Syntax error
        fixed_code_from_llm = "x = 1\nprint(x)"
        
        self.mock_llm.generate_json.return_value = {
            'fixed_code': fixed_code_from_llm,
            'changes_made': [
                {
                    'error_type': 'syntax_error',
                    'fix': 'Added missing closing paren',
                    'line': 2
                }
            ]
        }
        
        result_code, changes, success = self.validator.validate_and_fix(
            code=invalid_code,
            context="Test code",
            max_retries=2
        )
        
        self.assertEqual(result_code, fixed_code_from_llm)
        self.assertEqual(len(changes), 1)
        self.assertTrue(success)
        self.mock_llm.generate_json.assert_called_once()
        
        # Check that ValidationOutput schema was used
        call_kwargs = self.mock_llm.generate_json.call_args.kwargs
        self.assertEqual(call_kwargs['schema_name'], 'ValidationOutput')
    
    def test_max_retries_exhausted(self):
        """Test that max retries limit is respected"""
        invalid_code = "x = 1\nprint(x"
        still_invalid = "x = 1\nprint y"  # LLM returns still-invalid code
        
        self.mock_llm.generate_json.return_value = {
            'fixed_code': still_invalid,
            'changes_made': [
                {
                    'error_type': 'syntax_error',
                    'fix': 'Attempted fix',
                    'line': 2
                }
            ]
        }
        
        result_code, changes, success = self.validator.validate_and_fix(
            code=invalid_code,
            context="Test",
            max_retries=2
        )
        
        self.assertFalse(success)
        self.assertEqual(self.mock_llm.generate_json.call_count, 2)
    
    def test_validation_input_structure(self):
        """Test that validation input has correct structure"""
        invalid_code = "print(x"
        
        self.mock_llm.generate_json.return_value = {
            'fixed_code': "print(x)",
            'changes_made': []
        }
        
        self.validator.validate_and_fix(
            code=invalid_code,
            context="Test context",
            max_retries=1
        )
        
        # Check the input structure
        call_kwargs = self.mock_llm.generate_json.call_args.kwargs
        prompt_data = call_kwargs['prompt_data']
        
        self.assertIn('code', prompt_data)
        self.assertIn('context', prompt_data)
        self.assertIn('errors', prompt_data)
        self.assertIn('instructions', prompt_data)
        
        # Check errors structure
        self.assertGreater(len(prompt_data['errors']), 0)
        self.assertIn('error_type', prompt_data['errors'][0])


if __name__ == '__main__':
    unittest.main()
