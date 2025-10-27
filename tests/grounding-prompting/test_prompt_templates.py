#!/usr/bin/env python3
"""
Unit Tests for VTK Prompt Templates

Tests the production JSON-based prompt system:
- get_decomposition_instructions() (Sequential pipeline)
- get_generation_instructions() (Sequential pipeline)
- get_code_generation_instructions() (Simple queries)
- get_api_lookup_instructions() (API queries)
- get_explanation_instructions() (Concept queries)
- get_clarifying_question_instructions() (Clarifying queries)
- get_image_to_code_instructions() (FUTURE - Image→Code)
- get_code_to_image_instructions() (Code→Image)
- get_data_to_code_instructions() (Data→Code)
- get_code_to_data_instructions() (Code→Data)
"""

import unittest
import sys
from pathlib import Path

# Add grounding-prompting to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'grounding-prompting'))

from prompt_templates import VTKPromptTemplate


class TestDecompositionInstructions(unittest.TestCase):
    """Test decomposition instructions"""
    
    def setUp(self):
        self.template = VTKPromptTemplate()
    
    def test_returns_dict(self):
        """Should return a dictionary"""
        instructions = self.template.get_decomposition_instructions()
        self.assertIsInstance(instructions, dict)
    
    def test_has_required_keys(self):
        """Should have all required keys"""
        instructions = self.template.get_decomposition_instructions()
        required_keys = ['role', 'task', 'think_through', 'output_format_example', 'requirements', 'focus_areas']
        for key in required_keys:
            self.assertIn(key, instructions, f"Missing key: {key}")
    
    def test_role_is_string(self):
        """Role should be a non-empty string"""
        instructions = self.template.get_decomposition_instructions()
        self.assertIsInstance(instructions['role'], str)
        self.assertGreater(len(instructions['role']), 0)
        self.assertIn("VTK", instructions['role'])
    
    def test_task_is_string(self):
        """Task should be a non-empty string"""
        instructions = self.template.get_decomposition_instructions()
        self.assertIsInstance(instructions['task'], str)
        self.assertGreater(len(instructions['task']), 0)
    
    def test_think_through_is_list(self):
        """think_through should be a list of questions"""
        instructions = self.template.get_decomposition_instructions()
        self.assertIsInstance(instructions['think_through'], list)
        self.assertGreater(len(instructions['think_through']), 0)
        # Each item should be a string containing '?'
        for question in instructions['think_through']:
            self.assertIsInstance(question, str)
            self.assertIn('?', question, f"Not a question: {question}")
    
    def test_output_format_example_structure(self):
        """output_format_example should have expected structure"""
        instructions = self.template.get_decomposition_instructions()
        example = instructions['output_format_example']
        
        # Check required fields
        self.assertIn('understanding', example)
        self.assertIn('requires_visualization', example)
        self.assertIn('libraries_needed', example)
        self.assertIn('data_files', example)
        self.assertIn('steps', example)
        
        # Check types
        self.assertIsInstance(example['understanding'], str)
        self.assertIsInstance(example['requires_visualization'], bool)
        self.assertIsInstance(example['libraries_needed'], list)
        self.assertIsInstance(example['data_files'], list)
        self.assertIsInstance(example['steps'], list)
        
        # Check step structure
        if len(example['steps']) > 0:
            step = example['steps'][0]
            self.assertIn('step_number', step)
            self.assertIn('description', step)
            self.assertIn('search_query', step)
            self.assertIn('focus', step)
    
    def test_requirements_is_list(self):
        """requirements should be a list of strings"""
        instructions = self.template.get_decomposition_instructions()
        self.assertIsInstance(instructions['requirements'], list)
        self.assertGreater(len(instructions['requirements']), 0)
        for requirement in instructions['requirements']:
            self.assertIsInstance(requirement, str)
            self.assertTrue(requirement.startswith("DO NOT") or requirement.startswith("Return") or requirement.startswith("Include"))
    
    def test_focus_areas_is_string(self):
        """focus_areas should be a comma-separated string"""
        instructions = self.template.get_decomposition_instructions()
        self.assertIsInstance(instructions['focus_areas'], str)
        self.assertGreater(len(instructions['focus_areas']), 0)
        # Should contain common focus areas
        self.assertIn("geometry", instructions['focus_areas'])
        self.assertIn("rendering", instructions['focus_areas'])


class TestGenerationInstructions(unittest.TestCase):
    """Test generation instructions"""
    
    def setUp(self):
        self.template = VTKPromptTemplate()
    
    def test_returns_dict(self):
        """Should return a dictionary"""
        instructions = self.template.get_generation_instructions()
        self.assertIsInstance(instructions, dict)
    
    def test_has_required_keys(self):
        """Should have all required keys"""
        instructions = self.template.get_generation_instructions()
        required_keys = ['task', 'requirements', 'output_format']
        for key in required_keys:
            self.assertIn(key, instructions, f"Missing key: {key}")
    
    def test_task_is_string(self):
        """Task should be a non-empty string"""
        instructions = self.template.get_generation_instructions()
        self.assertIsInstance(instructions['task'], str)
        self.assertGreater(len(instructions['task']), 0)
        self.assertIn("code", instructions['task'].lower())
    
    def test_requirements_is_list(self):
        """requirements should be a list of strings"""
        instructions = self.template.get_generation_instructions()
        self.assertIsInstance(instructions['requirements'], list)
        self.assertGreater(len(instructions['requirements']), 5, "Should have multiple requirements")
        for requirement in instructions['requirements']:
            self.assertIsInstance(requirement, str)
            self.assertGreater(len(requirement), 0)
    
    def test_requirements_content(self):
        """Requirements should cover key aspects"""
        instructions = self.template.get_generation_instructions()
        requirements_text = " ".join(instructions['requirements'])
        
        # Check for key concepts
        key_concepts = ['import', 'code', 'documentation', 'cite', 'previous']
        for concept in key_concepts:
            self.assertIn(concept, requirements_text.lower(), f"Missing concept: {concept}")
    
    def test_output_format_is_string(self):
        """output_format should be a descriptive string"""
        instructions = self.template.get_generation_instructions()
        self.assertIsInstance(instructions['output_format'], str)
        self.assertGreater(len(instructions['output_format']), 0)
        # Should mention JSON structure
        self.assertIn("JSON", instructions['output_format'])
        self.assertIn("step_number", instructions['output_format'])
        self.assertIn("imports", instructions['output_format'])
        self.assertIn("code", instructions['output_format'])
        self.assertIn("citations", instructions['output_format'])


class TestPromptTemplateIntegration(unittest.TestCase):
    """Integration tests for prompt template usage"""
    
    def setUp(self):
        self.template = VTKPromptTemplate()
    
    def test_decomposition_input_structure(self):
        """Test building a complete decomposition input"""
        query = "How to create a cylinder in VTK?"
        
        decomposition_input = {
            "query": query,
            "instructions": self.template.get_decomposition_instructions()
        }
        
        # Verify structure
        self.assertIn("query", decomposition_input)
        self.assertIn("instructions", decomposition_input)
        self.assertEqual(decomposition_input["query"], query)
        self.assertIsInstance(decomposition_input["instructions"], dict)
    
    def test_generation_input_structure(self):
        """Test building a complete generation input"""
        generation_input = {
            "original_query": "Test query",
            "overall_understanding": "Test understanding",
            "overall_plan": {"total_steps": 2, "current_step_number": 1},
            "current_step": {"step_number": 1, "description": "Test step", "focus": "geometry"},
            "previous_steps": [],
            "documentation": [],
            "instructions": self.template.get_generation_instructions()
        }
        
        # Verify structure
        required_keys = ["original_query", "overall_understanding", "overall_plan", 
                        "current_step", "previous_steps", "documentation", "instructions"]
        for key in required_keys:
            self.assertIn(key, generation_input, f"Missing key: {key}")
        
        # Verify instructions
        self.assertIsInstance(generation_input["instructions"], dict)
        self.assertIn("task", generation_input["instructions"])
        self.assertIn("requirements", generation_input["instructions"])
    
    def test_multiple_instances_identical(self):
        """Multiple template instances should return identical instructions"""
        template1 = VTKPromptTemplate()
        template2 = VTKPromptTemplate()
        
        decomp1 = template1.get_decomposition_instructions()
        decomp2 = template2.get_decomposition_instructions()
        self.assertEqual(decomp1, decomp2)
        
        gen1 = template1.get_generation_instructions()
        gen2 = template2.get_generation_instructions()
        self.assertEqual(gen1, gen2)


class TestNewPromptMethods(unittest.TestCase):
    """Test new prompt methods added in Phase 1"""
    
    def setUp(self):
        self.template = VTKPromptTemplate()
    
    def test_api_lookup_instructions(self):
        """Test API lookup instructions"""
        instructions = self.template.get_api_lookup_instructions()
        
        # Basic structure
        self.assertIsInstance(instructions, dict)
        required_keys = ['role', 'task', 'output_format_example', 'requirements', 'grounding']
        for key in required_keys:
            self.assertIn(key, instructions)
        
        # Content checks
        self.assertIn("API", instructions['role'])
        self.assertIn("JSON", instructions['task'])
        
        # Output format
        example = instructions['output_format_example']
        self.assertEqual(example['response_type'], 'answer')
        self.assertEqual(example['content_type'], 'api')
        self.assertIn('explanation', example)
        self.assertIn('citations', example)
    
    def test_explanation_instructions(self):
        """Test explanation instructions"""
        instructions = self.template.get_explanation_instructions()
        
        self.assertIsInstance(instructions, dict)
        self.assertIn("concept", instructions['role'])
        
        example = instructions['output_format_example']
        self.assertEqual(example['content_type'], 'explanation')
        self.assertIn('explanation', example)
        self.assertIn('key_concepts', example)
    
    def test_clarifying_question_instructions(self):
        """Test clarifying question instructions"""
        instructions = self.template.get_clarifying_question_instructions()
        
        self.assertIsInstance(instructions, dict)
        example = instructions['output_format_example']
        self.assertEqual(example['response_type'], 'clarifying_question')
        self.assertIn('question', example)
        self.assertIn('reason', example)
    
    def test_code_to_image_instructions(self):
        """Test code-to-image instructions (implemented)"""
        instructions = self.template.get_code_to_image_instructions()
        
        self.assertIsInstance(instructions, dict)
        # Check docstring in method
        method_doc = self.template.get_code_to_image_instructions.__doc__
        self.assertIn("IMPLEMENTED", method_doc)
        
        example = instructions['output_format_example']
        self.assertEqual(example['content_type'], 'image')
        self.assertIn('baseline_images', example)
        self.assertIn('visual_description', example)
    
    def test_code_explanation_generation_instructions(self):
        """Test code explanation generation instructions"""
        instructions = self.template.get_code_explanation_generation_instructions()
        
        self.assertIsInstance(instructions, dict)
        self.assertIn("role", instructions)
        self.assertIn("task", instructions)
        self.assertIn("requirements", instructions)
        self.assertIn("output_format", instructions)
        
        # Check role mentions VTK expert
        self.assertIn("VTK", instructions['role'])
        self.assertIn("expert", instructions['role'])
        
        # Check task mentions generating explanation
        self.assertIn("explanation", instructions['task'])
        
        # Check requirements include key items
        requirements = instructions['requirements']
        self.assertIn("ExplanationEnrichmentOutput", str(requirements))
        self.assertTrue(any("VTK class" in str(req) for req in requirements))
        self.assertTrue(any("citation" in str(req) for req in requirements))
    
    def test_explanation_improvement_instructions(self):
        """Test explanation improvement instructions"""
        instructions = self.template.get_explanation_improvement_instructions()
        
        self.assertIsInstance(instructions, dict)
        self.assertIn("role", instructions)
        self.assertIn("task", instructions)
        self.assertIn("requirements", instructions)
        self.assertIn("output_format", instructions)
        
        # Check role mentions improving
        self.assertIn("improv", instructions['role'].lower())
        
        # Check task mentions enhancing
        self.assertIn("enhanc", instructions['task'].lower())
        
        # Check requirements include preserving original
        requirements = instructions['requirements']
        self.assertTrue(any("original" in str(req).lower() for req in requirements))
        self.assertIn("ExplanationEnrichmentOutput", str(requirements))
    
    def test_image_to_code_instructions(self):
        """Test image-to-code instructions (FUTURE)"""
        instructions = self.template.get_image_to_code_instructions()
        
        self.assertIsInstance(instructions, dict)
        # Check docstring in method, not dict
        method_doc = self.template.get_image_to_code_instructions.__doc__
        self.assertIn("FUTURE", method_doc)
        
        example = instructions['output_format_example']
        self.assertEqual(example['content_type'], 'code')
        self.assertIn('code', example)
        self.assertIn('image_analysis', example)
    
    def test_data_to_code_instructions(self):
        """Test data-to-code instructions (exploratory)"""
        instructions = self.template.get_data_to_code_instructions()
        
        self.assertIsInstance(instructions, dict)
        example = instructions['output_format_example']
        self.assertEqual(example['content_type'], 'code')
        self.assertIn('data_analysis', example)
        self.assertIn('suggested_techniques', example)
        self.assertIn('alternative_approaches', example)
        
        # Check alternative approaches structure
        alts = example['alternative_approaches']
        self.assertIsInstance(alts, list)
        if len(alts) > 0:
            alt = alts[0]
            self.assertIn('technique', alt)
            self.assertIn('description', alt)
            self.assertIn('vtk_classes', alt)
            self.assertIn('complexity', alt)
    
    def test_code_to_data_instructions(self):
        """Test code-to-data instructions (find data files)"""
        instructions = self.template.get_code_to_data_instructions()
        
        self.assertIsInstance(instructions, dict)
        example = instructions['output_format_example']
        self.assertEqual(example['content_type'], 'data')
        self.assertIn('data_files', example)
        self.assertIn('code_requirements', example if 'code_requirements' in example else instructions['output_format_example'])
        
        # Check data files structure
        files = example['data_files']
        self.assertIsInstance(files, list)
        if len(files) > 0:
            file_info = files[0]
            self.assertIn('filename', file_info)
            self.assertIn('download_url', file_info)
    
    def test_all_methods_return_dicts(self):
        """All prompt methods should return dicts"""
        methods = [
            'get_decomposition_instructions',
            'get_generation_instructions',
            'get_code_generation_instructions',
            'get_api_lookup_instructions',
            'get_explanation_instructions',
            'get_clarifying_question_instructions',
            'get_image_to_code_instructions',
            'get_code_to_image_instructions',
            'get_data_to_code_instructions',
            'get_code_to_data_instructions',
        ]
        
        for method_name in methods:
            with self.subTest(method=method_name):
                method = getattr(self.template, method_name)
                result = method()
                self.assertIsInstance(result, dict, f"{method_name} should return dict")
    
    def test_all_methods_have_required_structure(self):
        """All prompt methods should have basic required keys"""
        methods = [
            'get_api_lookup_instructions',
            'get_explanation_instructions',
            'get_data_to_code_instructions',
            'get_code_to_data_instructions',
            'get_code_to_image_instructions',
        ]
        
        for method_name in methods:
            with self.subTest(method=method_name):
                method = getattr(self.template, method_name)
                result = method()
                
                # All should have these keys
                self.assertIn('role', result)
                self.assertIn('task', result)
                self.assertIn('output_format_example', result)
                self.assertIn('requirements', result)
                self.assertIn('grounding', result)
                
                # Requirements and grounding should be lists
                self.assertIsInstance(result['requirements'], list)
                self.assertIsInstance(result['grounding'], list)


if __name__ == '__main__':
    unittest.main(verbosity=2)
