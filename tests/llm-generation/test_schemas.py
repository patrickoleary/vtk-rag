#!/usr/bin/env python3
"""
Unit Tests for JSON Schemas

Tests schema validation, serialization, and deserialization.
"""

import sys
from pathlib import Path
import unittest
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))

from unittest.mock import Mock, patch

from schemas import (
    DecompositionInput, DecompositionOutput, Step,
    GenerationInput, GenerationOutput, DocumentationChunk,
    PreviousStepResult, CurrentStepInfo, OverallPlan,
    ValidationInput, ValidationOutput, ValidationError, ValidationFix,
    FinalResult, FinalCode, Explanation, StepResult,
    APILookupOutput, ExplanationOutput, DataToCodeOutput, CodeToDataOutput,
    ModificationDecompositionOutput, CodeModificationOutput, CodeRefinementResult,
    validate_decomposition_output, validate_generation_output, validate_validation_output,
    validate_api_lookup_output, validate_explanation_output,
    validate_data_to_code_output, validate_code_to_data_output,
    validate_modification_decomposition_output, validate_code_modification_output
)

from llm_client import LLMClient, LLMProvider


class TestDecompositionSchemas(unittest.TestCase):
    """Test decomposition phase schemas"""
    
    def test_decomposition_output_valid(self):
        """Test valid decomposition output"""
        data = {
            "understanding": "User wants to create a cylinder",
            "requires_visualization": True,
            "libraries_needed": ["vtk"],
            "data_files": [],
            "steps": [
                {
                    "step_number": 1,
                    "description": "Create cylinder source",
                    "search_query": "VTK vtkCylinderSource",
                    "focus": "geometry"
                }
            ]
        }
        
        self.assertTrue(validate_decomposition_output(data))
        
        # Test deserialization
        decomp = DecompositionOutput.from_dict(data)
        self.assertEqual(decomp.understanding, "User wants to create a cylinder")
        self.assertEqual(len(decomp.steps), 1)
        
        # Test step conversion
        steps = decomp.get_steps()
        self.assertIsInstance(steps[0], Step)
        self.assertEqual(steps[0].step_number, 1)
    
    def test_decomposition_output_missing_fields(self):
        """Test decomposition output with missing fields"""
        data = {
            "understanding": "Test",
            "requires_visualization": True
            # Missing: libraries_needed, data_files, steps
        }
        
        self.assertFalse(validate_decomposition_output(data))
    
    def test_decomposition_input_serialization(self):
        """Test decomposition input can be serialized to JSON"""
        decomp_input = DecompositionInput(
            query="Create a cylinder",
            instructions={
                "role": "Expert",
                "task": "Analyze query",
                "think_through": ["Step 1", "Step 2"],
                "requirements": ["Do not hallucinate"]
            }
        )
        
        # Test to_dict
        data = decomp_input.to_dict()
        self.assertIn("query", data)
        self.assertIn("instructions", data)
        
        # Test to_json
        json_str = decomp_input.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["query"], "Create a cylinder")


class TestGenerationSchemas(unittest.TestCase):
    """Test generation phase schemas"""
    
    def test_generation_output_valid(self):
        """Test valid generation output"""
        data = {
            "step_number": 1,
            "understanding": "Creates cylinder source",
            "imports": ["from vtkmodules.vtkFiltersSources import vtkCylinderSource"],
            "code": "cylinder = vtkCylinderSource()",
            "citations": [1, 2]
        }
        
        self.assertTrue(validate_generation_output(data))
        
        # Test deserialization
        gen_output = GenerationOutput.from_dict(data)
        self.assertEqual(gen_output.step_number, 1)
        self.assertEqual(len(gen_output.imports), 1)
        self.assertEqual(len(gen_output.citations), 2)
    
    def test_generation_output_missing_fields(self):
        """Test generation output with missing fields"""
        data = {
            "step_number": 1,
            "understanding": "Test"
            # Missing: imports, code, citations
        }
        
        self.assertFalse(validate_generation_output(data))
    
    def test_generation_input_serialization(self):
        """Test generation input can be serialized"""
        gen_input = GenerationInput(
            original_query="Create cylinder",
            overall_understanding="User wants cylinder",
            overall_plan={
                "total_steps": 2,
                "current_step_number": 1,
                "steps": []
            },
            current_step={
                "step_number": 1,
                "description": "Create source",
                "focus": "geometry"
            },
            previous_steps=[],
            documentation=[],
            instructions={
                "task": "Generate code",
                "requirements": [],
                "output_format": "JSON"
            }
        )
        
        # Test to_dict
        data = gen_input.to_dict()
        self.assertIn("original_query", data)
        self.assertIn("overall_plan", data)
        
        # Test to_json
        json_str = gen_input.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["original_query"], "Create cylinder")


class TestValidationSchemas(unittest.TestCase):
    """Test validation phase schemas"""
    
    def test_validation_output_valid(self):
        """Test valid validation output"""
        data = {
            "fixed_code": "corrected code here",
            "changes_made": [
                {
                    "error_type": "NameError",
                    "fix": "Added import",
                    "line": 1
                }
            ]
        }
        
        self.assertTrue(validate_validation_output(data))
        
        # Test deserialization
        val_output = ValidationOutput.from_dict(data)
        self.assertEqual(val_output.fixed_code, "corrected code here")
        self.assertEqual(len(val_output.changes_made), 1)
    
    def test_validation_output_missing_fields(self):
        """Test validation output with missing fields"""
        data = {
            "fixed_code": "code"
            # Missing: changes_made
        }
        
        self.assertFalse(validate_validation_output(data))
    
    def test_validation_input_serialization(self):
        """Test validation input can be serialized"""
        val_input = ValidationInput(
            task="Fix errors",
            original_query="Create cylinder",
            generated_code="broken code",
            validation_errors=[
                {
                    "type": "SyntaxError",
                    "message": "Invalid syntax",
                    "line": 5,
                    "context": "bad line"
                }
            ],
            instructions={
                "task": "Fix code",
                "requirements": [],
                "output_format": "JSON"
            }
        )
        
        # Test to_dict
        data = val_input.to_dict()
        self.assertIn("generated_code", data)
        self.assertIn("validation_errors", data)
        
        # Test to_json
        json_str = val_input.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(len(parsed["validation_errors"]), 1)


class TestFinalResultSchema(unittest.TestCase):
    """Test final result schema"""
    
    def test_final_result_serialization(self):
        """Test final result can be serialized"""
        final_result = FinalResult(
            query="Create cylinder",
            understanding="User wants cylinder",
            requires_visualization=True,
            libraries_needed=["vtk"],
            data_files=[],
            steps=[
                {
                    "step_number": 1,
                    "description": "Create source",
                    "understanding": "Creates cylinder",
                    "imports": ["import vtk"],
                    "code": "cylinder = vtkCylinderSource()",
                    "citations": [1],
                    "chunks_used": ["chunk1"]
                }
            ],
            final_code={
                "imports": "import vtk",
                "body": "cylinder = vtkCylinderSource()",
                "complete": "import vtk\n\ncylinder = vtkCylinderSource()"
            },
            explanation={
                "overview": "Creates cylinder",
                "imports": ["import vtk"],
                "steps": [],
                "formatted": "Step 1: ..."
            }
        )
        
        # Test to_dict
        data = final_result.to_dict()
        self.assertIn("query", data)
        self.assertIn("final_code", data)
        self.assertIn("explanation", data)
        
        # Test to_json
        json_str = final_result.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["query"], "Create cylinder")
        self.assertEqual(len(parsed["steps"]), 1)


class TestNewSchemaDefinitions(unittest.TestCase):
    """Test that new schemas are properly defined (Phase 1-3)"""
    
    def setUp(self):
        """Create mock LLM client"""
        self.mock_config = Mock()
        self.mock_config.provider = LLMProvider.OPENAI
        self.mock_config.model = "gpt-4"
        self.mock_config.api_key = "test"
        
        self.client = LLMClient(config=self.mock_config)
    
    def test_api_lookup_schema_defined(self):
        """APILookupOutput schema should be defined"""
        schema = self.client._get_schema_details("APILookupOutput")
        
        self.assertIsInstance(schema, str)
        self.assertIn("response_type", schema)
        self.assertIn("content_type", schema)
        self.assertIn("api", schema)
        self.assertIn("explanation", schema)
        self.assertIn("parameters", schema)
    
    def test_explanation_schema_defined(self):
        """ExplanationOutput schema should be defined"""
        schema = self.client._get_schema_details("ExplanationOutput")
        
        self.assertIsInstance(schema, str)
        self.assertIn("response_type", schema)
        self.assertIn("explanation", schema)
        self.assertIn("key_concepts", schema)
    
    def test_data_to_code_schema_defined(self):
        """DataToCodeOutput schema should be defined"""
        schema = self.client._get_schema_details("DataToCodeOutput")
        
        self.assertIsInstance(schema, str)
        self.assertIn("data_analysis", schema)
        self.assertIn("suggested_techniques", schema)
        self.assertIn("alternative_approaches", schema)
        self.assertIn("code", schema)
    
    def test_code_to_data_schema_defined(self):
        """CodeToDataOutput schema should be defined"""
        schema = self.client._get_schema_details("CodeToDataOutput")
        
        self.assertIsInstance(schema, str)
        self.assertIn("data_files", schema)
        self.assertIn("code_requirements", schema)
        self.assertIn("download_url", schema)
    
    def test_all_schemas_have_confidence(self):
        """All new schemas should have confidence field"""
        schema_names = [
            "APILookupOutput",
            "ExplanationOutput",
            "DataToCodeOutput",
            "CodeToDataOutput"
        ]
        
        for schema_name in schema_names:
            with self.subTest(schema=schema_name):
                schema = self.client._get_schema_details(schema_name)
                self.assertIn("confidence", schema)
                self.assertIn("high|medium|low", schema)
    
    def test_all_schemas_have_citations(self):
        """All new schemas should have citations field"""
        schema_names = [
            "APILookupOutput",
            "ExplanationOutput",
            "DataToCodeOutput",
            "CodeToDataOutput"
        ]
        
        for schema_name in schema_names:
            with self.subTest(schema=schema_name):
                schema = self.client._get_schema_details(schema_name)
                self.assertIn("citations", schema)


class TestNewSchemaValidation(unittest.TestCase):
    """Test schema validation for new query types (Phase 1-3)"""
    
    def setUp(self):
        """Create mock LLM client"""
        self.mock_config = Mock()
        self.mock_config.provider = LLMProvider.OPENAI
        self.mock_config.model = "gpt-4"
        self.mock_config.api_key = "test"
        
        self.client = LLMClient(config=self.mock_config)
    
    def test_api_lookup_valid_schema(self):
        """Valid APILookupOutput should pass validation"""
        valid_data = {
            "response_type": "answer",
            "content_type": "api",
            "explanation": "Test explanation",
            "confidence": "high",
            "citations": [{"number": 1, "reason": "test"}]
        }
        
        result = self.client._validate_json_schema(valid_data, "APILookupOutput")
        self.assertTrue(result)
    
    def test_api_lookup_invalid_schema(self):
        """Invalid APILookupOutput should fail validation"""
        invalid_data = {
            "response_type": "answer",
            # Missing required fields
        }
        
        result = self.client._validate_json_schema(invalid_data, "APILookupOutput")
        self.assertFalse(result)
    
    def test_explanation_valid_schema(self):
        """Valid ExplanationOutput should pass validation"""
        valid_data = {
            "response_type": "answer",
            "content_type": "explanation",
            "explanation": "Test explanation",
            "confidence": "high",
            "citations": []
        }
        
        result = self.client._validate_json_schema(valid_data, "ExplanationOutput")
        self.assertTrue(result)
    
    def test_data_to_code_valid_schema(self):
        """Valid DataToCodeOutput should pass validation"""
        valid_data = {
            "response_type": "answer",
            "content_type": "code",
            "code": "print('test')",
            "explanation": "Test explanation",
            "confidence": "high",
            "citations": []
        }
        
        result = self.client._validate_json_schema(valid_data, "DataToCodeOutput")
        self.assertTrue(result)
    
    def test_code_to_data_valid_schema(self):
        """Valid CodeToDataOutput should pass validation"""
        valid_data = {
            "response_type": "answer",
            "content_type": "data",
            "explanation": "Test explanation",
            "confidence": "high",
            "citations": []
        }
        
        result = self.client._validate_json_schema(valid_data, "CodeToDataOutput")
        self.assertTrue(result)
    
    def test_validation_warns_for_unknown_schema(self):
        """Should log warning for truly unknown schema"""
        with self.assertLogs('llm_client', level='WARNING') as cm:
            self.client._validate_json_schema(
                {"test": "data"},
                "UnknownSchemaType"
            )
        
        # Should have warning about no validator
        self.assertTrue(any("No validator found" in msg for msg in cm.output))


class TestNewSchemaIntegration(unittest.TestCase):
    """Test new schemas work with generate_json (Phase 1-3)"""
    
    def setUp(self):
        """Create mock LLM client"""
        self.mock_config = Mock()
        self.mock_config.provider = LLMProvider.OPENAI
        self.mock_config.model = "gpt-4"
        self.mock_config.api_key = "test"
        self.mock_config.temperature = 0.1
        self.mock_config.max_tokens = 2000
        
        self.client = LLMClient(config=self.mock_config)
    
    @patch.object(LLMClient, '_generate_openai')
    def test_api_lookup_schema_in_generate_json(self, mock_generate):
        """APILookupOutput schema should work with generate_json"""
        # Mock response with VALID schema data
        mock_generate.return_value = '''{
            "response_type": "answer",
            "content_type": "api",
            "explanation": "Test explanation",
            "confidence": "high",
            "citations": [{"number": 1, "reason": "test"}]
        }'''
        
        prompt_data = {
            "instructions": {"role": "test"},
            "query": "What does SetMapper do?"
        }
        
        # Should not raise exception
        result = self.client.generate_json(
            prompt_data=prompt_data,
            schema_name="APILookupOutput"
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['response_type'], 'answer')
        self.assertEqual(result['content_type'], 'api')
        
        # Check that schema was used in system message
        call_args = mock_generate.call_args
        system_msg = call_args[0][0]  # First positional arg
        self.assertIn("APILookupOutput", system_msg)


class TestExplanationEnrichmentSchema(unittest.TestCase):
    """Test ExplanationEnrichmentOutput schema"""
    
    def test_explanation_enrichment_schema_defined(self):
        """ExplanationEnrichmentOutput schema should be defined"""
        from schemas import ExplanationEnrichmentOutput
        
        # Should have required fields
        self.assertTrue(hasattr(ExplanationEnrichmentOutput, '__dataclass_fields__'))
        fields = ExplanationEnrichmentOutput.__dataclass_fields__
        
        self.assertIn('improved_explanation', fields)
        self.assertIn('confidence', fields)
        self.assertIn('citations', fields)
        self.assertIn('key_points', fields)
        self.assertIn('vtk_classes_explained', fields)
    
    def test_explanation_enrichment_validation(self):
        """ExplanationEnrichmentOutput validation should work"""
        from schemas import validate_explanation_enrichment_output
        
        # Valid data
        valid_data = {
            'improved_explanation': 'Test explanation',
            'confidence': 'high',
            'citations': []
        }
        self.assertTrue(validate_explanation_enrichment_output(valid_data))
        
        # Missing required field
        invalid_data = {
            'improved_explanation': 'Test',
            # Missing confidence
            'citations': []
        }
        self.assertFalse(validate_explanation_enrichment_output(invalid_data))
    
    def test_explanation_enrichment_from_dict(self):
        """ExplanationEnrichmentOutput should construct from dict"""
        from schemas import ExplanationEnrichmentOutput
        
        data = {
            'improved_explanation': 'Detailed explanation',
            'confidence': 'high',
            'citations': [{'number': 1, 'reason': 'test'}],
            'key_points': ['Point 1', 'Point 2'],
            'vtk_classes_explained': [{'name': 'vtkTest', 'purpose': 'Testing'}]
        }
        
        obj = ExplanationEnrichmentOutput.from_dict(data)
        
        self.assertEqual(obj.improved_explanation, 'Detailed explanation')
        self.assertEqual(obj.confidence, 'high')
        self.assertEqual(len(obj.citations), 1)
        self.assertEqual(len(obj.key_points), 2)
        self.assertEqual(len(obj.vtk_classes_explained), 1)
    
    def test_explanation_enrichment_to_dict(self):
        """ExplanationEnrichmentOutput should convert to dict"""
        from schemas import ExplanationEnrichmentOutput
        
        obj = ExplanationEnrichmentOutput(
            improved_explanation='Test',
            confidence='high',
            citations=[],
            key_points=['A', 'B'],
            vtk_classes_explained=None
        )
        
        result = obj.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['improved_explanation'], 'Test')
        self.assertEqual(result['confidence'], 'high')
        self.assertEqual(result['key_points'], ['A', 'B'])


class TestRefinementSchemas(unittest.TestCase):
    """Test code refinement schemas"""
    
    def test_modification_decomposition_schema_defined(self):
        """ModificationDecompositionOutput schema should be defined"""
        # Should have required fields
        self.assertTrue(hasattr(ModificationDecompositionOutput, '__dataclass_fields__'))
        fields = ModificationDecompositionOutput.__dataclass_fields__
        
        self.assertIn('understanding', fields)
        self.assertIn('modification_steps', fields)
        self.assertIn('preserved_elements', fields)
    
    def test_modification_decomposition_validation(self):
        """ModificationDecompositionOutput validation should work"""
        # Valid data
        valid_data = {
            'understanding': 'User wants to change color',
            'modification_steps': [
                {
                    'step_number': 1,
                    'description': 'Change color to blue',
                    'requires_retrieval': False
                }
            ],
            'preserved_elements': ['resolution', 'size']
        }
        self.assertTrue(validate_modification_decomposition_output(valid_data))
        
        # Missing required field
        invalid_data = {
            'understanding': 'Test',
            # Missing modification_steps and preserved_elements
        }
        self.assertFalse(validate_modification_decomposition_output(invalid_data))
    
    def test_modification_decomposition_from_dict(self):
        """ModificationDecompositionOutput should construct from dict"""
        data = {
            'understanding': 'Increase resolution and change color',
            'modification_steps': [
                {
                    'step_number': 1,
                    'description': 'Increase resolution to 50',
                    'requires_retrieval': False
                },
                {
                    'step_number': 2,
                    'description': 'Change color to blue',
                    'requires_retrieval': False
                }
            ],
            'preserved_elements': ['variable names', 'structure']
        }
        
        obj = ModificationDecompositionOutput.from_dict(data)
        
        self.assertEqual(obj.understanding, 'Increase resolution and change color')
        self.assertEqual(len(obj.modification_steps), 2)
        self.assertEqual(len(obj.preserved_elements), 2)
    
    def test_code_modification_schema_defined(self):
        """CodeModificationOutput schema should be defined"""
        self.assertTrue(hasattr(CodeModificationOutput, '__dataclass_fields__'))
        fields = CodeModificationOutput.__dataclass_fields__
        
        self.assertIn('modifications', fields)
        self.assertIn('updated_code', fields)
        self.assertIn('new_imports', fields)
        self.assertIn('preserved_structure', fields)
        self.assertIn('diff_summary', fields)
    
    def test_code_modification_validation(self):
        """CodeModificationOutput validation should work"""
        # Valid data
        valid_data = {
            'modifications': [
                {
                    'step_number': 1,
                    'modification': 'Changed resolution',
                    'explanation': 'Increased from 8 to 50',
                    'code_changed': 'cylinder.SetResolution(50)',
                    'code_added': '',
                    'variable_affected': 'cylinder'
                }
            ],
            'updated_code': 'cylinder.SetResolution(50)',
            'new_imports': [],
            'preserved_structure': True,
            'diff_summary': 'Changed resolution'
        }
        self.assertTrue(validate_code_modification_output(valid_data))
        
        # Missing required field
        invalid_data = {
            'modifications': [],
            # Missing updated_code
        }
        self.assertFalse(validate_code_modification_output(invalid_data))
    
    def test_code_modification_from_dict(self):
        """CodeModificationOutput should construct from dict"""
        data = {
            'modifications': [
                {
                    'step_number': 1,
                    'modification': 'Added color',
                    'explanation': 'Set color to blue',
                    'code_changed': '',
                    'code_added': 'actor.GetProperty().SetColor(0, 0, 1)',
                    'variable_affected': 'actor'
                }
            ],
            'updated_code': 'complete code here',
            'new_imports': ['from vtkmodules.vtkCommonColor import vtkNamedColors'],
            'preserved_structure': True,
            'diff_summary': 'Added blue color'
        }
        
        obj = CodeModificationOutput.from_dict(data)
        
        self.assertEqual(len(obj.modifications), 1)
        self.assertEqual(obj.updated_code, 'complete code here')
        self.assertEqual(len(obj.new_imports), 1)
        self.assertTrue(obj.preserved_structure)
    
    def test_code_refinement_result_defined(self):
        """CodeRefinementResult schema should be defined"""
        self.assertTrue(hasattr(CodeRefinementResult, '__dataclass_fields__'))
        fields = CodeRefinementResult.__dataclass_fields__
        
        self.assertIn('response_type', fields)
        self.assertIn('content_type', fields)
        self.assertIn('query', fields)
        self.assertIn('original_code', fields)
        self.assertIn('code', fields)
        self.assertIn('explanation', fields)
        self.assertIn('modifications', fields)
        self.assertIn('citations', fields)
        self.assertIn('confidence', fields)
        self.assertIn('diff', fields)
    
    def test_code_refinement_result_to_dict(self):
        """CodeRefinementResult should convert to dict"""
        obj = CodeRefinementResult(
            response_type='answer',
            content_type='code_refinement',
            query='Make it blue',
            original_code='original code',
            code='modified code',
            explanation='Changed color to blue',
            modifications=[],
            new_imports=[],
            citations=[],
            chunk_ids_used=[],
            confidence='high',
            diff='diff content'
        )
        
        result = obj.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['response_type'], 'answer')
        self.assertEqual(result['content_type'], 'code_refinement')
        self.assertEqual(result['query'], 'Make it blue')
        self.assertEqual(result['confidence'], 'high')
    
    def test_refinement_schemas_in_llm_client(self):
        """Refinement schemas should be registered in LLM client"""
        mock_config = Mock()
        mock_config.provider = LLMProvider.OPENAI
        mock_config.model = "gpt-4"
        mock_config.api_key = "test"
        
        client = LLMClient(config=mock_config)
        
        # Test ModificationDecompositionOutput
        schema = client._get_schema_details("ModificationDecompositionOutput")
        self.assertIsInstance(schema, str)
        self.assertIn("understanding", schema)
        self.assertIn("modification_steps", schema)
        self.assertIn("preserved_elements", schema)
        
        # Test CodeModificationOutput
        schema = client._get_schema_details("CodeModificationOutput")
        self.assertIsInstance(schema, str)
        self.assertIn("modifications", schema)
        self.assertIn("updated_code", schema)
        self.assertIn("new_imports", schema)


if __name__ == '__main__':
    unittest.main()
