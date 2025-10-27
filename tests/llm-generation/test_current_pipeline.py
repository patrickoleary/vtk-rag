#!/usr/bin/env python3
"""
Basic tests for current pipeline (before JSON refactor)

These tests verify the current implementation still works.
More comprehensive tests will be added in Sprint 2-3 after refactoring.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'retrieval-pipeline'))

from sequential_pipeline import SequentialPipeline, QueryStep


class TestSequentialPipelineBasics(unittest.TestCase):
    """Basic tests for sequential pipeline"""
    
    def test_query_step_creation(self):
        """Test creating a QueryStep"""
        step = QueryStep(
            step_number=1,
            description="Test step",
            query="test query",
            focus="test"
        )
        
        self.assertEqual(step.step_number, 1)
        self.assertEqual(step.description, "Test step")
        self.assertEqual(step.query, "test query")
        self.assertEqual(step.focus, "test")
    
    @patch('sequential_pipeline.TaskSpecificRetriever')
    @patch('sequential_pipeline.VTKRAGGenerator')
    @patch('sequential_pipeline.LLMClient')
    def test_pipeline_initialization(self, mock_llm, mock_gen, mock_ret):
        """Test pipeline can be initialized"""
        # Mock the dependencies
        mock_llm.from_env.return_value = Mock()
        
        pipeline = SequentialPipeline(
            use_llm_decomposition=False,  # Use heuristic for now
            enable_validation=False
        )
        
        self.assertIsNotNone(pipeline)
        self.assertFalse(pipeline.use_llm_decomposition)
    
    @patch('sequential_pipeline.TaskSpecificRetriever')
    @patch('sequential_pipeline.VTKRAGGenerator')
    @patch('sequential_pipeline.LLMClient')
    def test_heuristic_decomposition(self, mock_llm, mock_gen, mock_ret):
        """Test heuristic decomposition produces steps"""
        mock_llm.from_env.return_value = Mock()
        
        pipeline = SequentialPipeline(
            use_llm_decomposition=False,
            enable_validation=False
        )
        
        steps = pipeline.decompose_query_heuristic(
            "Create a cylinder and render it"
        )
        
        self.assertIsInstance(steps, list)
        self.assertTrue(len(steps) > 0)
        self.assertIsInstance(steps[0], QueryStep)
    
    @patch('sequential_pipeline.TaskSpecificRetriever')
    @patch('sequential_pipeline.VTKRAGGenerator')
    @patch('sequential_pipeline.LLMClient')
    def test_llm_decomposition_uses_json(self, mock_llm, mock_gen, mock_ret):
        """Test LLM decomposition uses JSON-based communication"""
        # Mock the LLM client
        mock_llm_instance = Mock()
        mock_llm_instance.generate_json = Mock(return_value={
            "understanding": "Create and render a cylinder",
            "requires_visualization": True,
            "libraries_needed": ["vtk"],
            "data_files": [],
            "steps": [
                {
                    "step_number": 1,
                    "description": "Create cylinder source",
                    "search_query": "VTK create cylinder source",
                    "focus": "geometry"
                },
                {
                    "step_number": 2,
                    "description": "Create mapper",
                    "search_query": "VTK create mapper",
                    "focus": "rendering"
                }
            ]
        })
        mock_llm.return_value = mock_llm_instance
        
        pipeline = SequentialPipeline(
            use_llm_decomposition=True,
            enable_validation=False
        )
        
        # Replace the llm_client with our mock
        pipeline.llm_client = mock_llm_instance
        
        steps = pipeline.decompose_query_llm("Create a cylinder")
        
        # Verify generate_json was called
        mock_llm_instance.generate_json.assert_called_once()
        call_kwargs = mock_llm_instance.generate_json.call_args.kwargs
        
        # Verify correct schema name
        self.assertEqual(call_kwargs['schema_name'], 'DecompositionOutput')
        
        # Verify structured input with query and instructions
        prompt_data = call_kwargs['prompt_data']
        self.assertIn('query', prompt_data)
        self.assertIn('instructions', prompt_data)
        self.assertEqual(prompt_data['query'], "Create a cylinder")
        
        # Verify output
        self.assertEqual(len(steps), 2)
        self.assertIsInstance(steps[0], QueryStep)
        self.assertEqual(steps[0].step_number, 1)
        self.assertEqual(steps[0].description, "Create cylinder source")
    
    @patch('sequential_pipeline.TaskSpecificRetriever')
    @patch('sequential_pipeline.VTKRAGGenerator')
    @patch('sequential_pipeline.LLMClient')
    def test_assemble_final_result_deduplicates_imports(self, mock_llm, mock_gen, mock_ret):
        """Test that _assemble_final_result deduplicates imports"""
        mock_llm.return_value = Mock()
        
        pipeline = SequentialPipeline(
            use_llm_decomposition=False,
            enable_validation=False
        )
        
        steps = [
            QueryStep(1, "Step 1", "query1", "geometry"),
            QueryStep(2, "Step 2", "query2", "rendering")
        ]
        
        step_solutions = [
            {
                'step_number': 1,
                'understanding': 'Creates cylinder',
                'imports': ['import vtk', 'from vtkmodules.vtkFiltersSources import vtkCylinderSource'],
                'code': 'cylinder = vtkCylinderSource()',
                'citations': [1]
            },
            {
                'step_number': 2,
                'understanding': 'Creates mapper',
                'imports': ['import vtk', 'from vtkmodules.vtkRenderingCore import vtkPolyDataMapper'],  # 'import vtk' is duplicate
                'code': 'mapper = vtkPolyDataMapper()',
                'citations': [2]
            }
        ]
        
        result = pipeline._assemble_final_result(
            query="Test query",
            steps=steps,
            step_solutions=step_solutions,
            chunk_ids=['chunk1', 'chunk2']
        )
        
        # Check that imports are deduplicated
        # Get only the imports section (before the first blank line after imports)
        lines = result.code.split('\n')
        import_lines = []
        for line in lines:
            if line.startswith(('import ', 'from ')):
                import_lines.append(line)
            elif line.strip() == '' and import_lines:
                break
        
        import_vtk_count = sum(1 for line in import_lines if line.strip() == 'import vtk')
        self.assertEqual(import_vtk_count, 1, "Duplicate imports should be removed")
        
        # Check that we have 3 import statements total (deduplicated)
        self.assertEqual(len(import_lines), 3, "Should have 3 unique imports")
        
        # Check that both codes are present
        self.assertIn('cylinder = vtkCylinderSource()', result.code)
        self.assertIn('mapper = vtkPolyDataMapper()', result.code)
    
    @patch('sequential_pipeline.TaskSpecificRetriever')
    @patch('sequential_pipeline.VTKRAGGenerator')
    @patch('sequential_pipeline.LLMClient')
    def test_assemble_final_result_structure(self, mock_llm, mock_gen, mock_ret):
        """Test that _assemble_final_result has correct structure: imports then code"""
        mock_llm.return_value = Mock()
        
        pipeline = SequentialPipeline(
            use_llm_decomposition=False,
            enable_validation=False
        )
        
        steps = [QueryStep(1, "Step 1", "query1", "geometry")]
        
        step_solutions = [
            {
                'step_number': 1,
                'understanding': 'Creates cylinder',
                'imports': ['import vtk'],
                'code': 'cylinder = vtkCylinderSource()',
                'citations': [1]
            }
        ]
        
        result = pipeline._assemble_final_result(
            query="Test query",
            steps=steps,
            step_solutions=step_solutions,
            chunk_ids=['chunk1']
        )
        
        # Check structure: imports before code
        code_lines = result.code.split('\n')
        import_line_idx = next(i for i, line in enumerate(code_lines) if line.startswith('import'))
        code_line_idx = next(i for i, line in enumerate(code_lines) if 'cylinder' in line)
        
        self.assertLess(import_line_idx, code_line_idx, "Imports should come before code")
    
    @patch('sequential_pipeline.TaskSpecificRetriever')
    @patch('sequential_pipeline.VTKRAGGenerator')
    @patch('sequential_pipeline.LLMClient')
    def test_assemble_final_result_empty_solutions(self, mock_llm, mock_gen, mock_ret):
        """Test that _assemble_final_result handles empty solutions"""
        mock_llm.return_value = Mock()
        
        pipeline = SequentialPipeline(
            use_llm_decomposition=False,
            enable_validation=False
        )
        
        result = pipeline._assemble_final_result(
            query="Test query",
            steps=[],
            step_solutions=[],
            chunk_ids=[]
        )
        
        self.assertEqual(result.code, "")
        self.assertEqual(result.explanation, "No solutions generated")


class TestGeneratorBasics(unittest.TestCase):
    """Basic tests for VTK RAG generator"""
    
    @patch('generator.LLMClient')
    def test_generator_initialization(self, mock_llm):
        """Test generator can be initialized"""
        from generator import VTKRAGGenerator
        
        mock_llm_instance = Mock()
        mock_llm.from_env.return_value = mock_llm_instance
        
        generator = VTKRAGGenerator(
            require_citations=True,
            strict_grounding=True,
            validate_code=False
        )
        
        self.assertIsNotNone(generator)
        self.assertTrue(generator.require_citations)
        self.assertTrue(generator.strict_grounding)


if __name__ == '__main__':
    unittest.main()
