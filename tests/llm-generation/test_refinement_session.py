#!/usr/bin/env python3
"""
Unit Tests for Refinement Session (Undo/Rollback)

Tests the RefinementSession class and pipeline integration.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))

from refinement_session import RefinementSession, RefinementVersion
from sequential_pipeline import SequentialPipeline


class TestRefinementSession(unittest.TestCase):
    """Test RefinementSession class"""
    
    def setUp(self):
        """Create a session for testing"""
        self.initial_code = "cylinder = vtkCylinderSource()"
        self.session = RefinementSession(self.initial_code)
    
    def test_session_creation(self):
        """Should create session with initial version"""
        self.assertIsNotNone(self.session.session_id)
        self.assertEqual(self.session.get_version_count(), 1)
        self.assertEqual(self.session.get_current_version(), 0)
        self.assertEqual(self.session.get_current_code(), self.initial_code)
    
    def test_session_id_custom(self):
        """Should accept custom session ID"""
        session = RefinementSession(self.initial_code, session_id="test-123")
        self.assertEqual(session.session_id, "test-123")
    
    def test_add_refinement(self):
        """Should add refinement and increment version"""
        result = {
            'code': 'cylinder.SetResolution(50)',
            'modifications': [{'step': 1}],
            'explanation': 'Increased resolution',
            'diff': 'some diff'
        }
        
        version = self.session.add_refinement("Increase resolution", result)
        
        self.assertEqual(version, 1)
        self.assertEqual(self.session.get_version_count(), 2)
        self.assertEqual(self.session.get_current_version(), 1)
        self.assertEqual(self.session.get_current_code(), 'cylinder.SetResolution(50)')
    
    def test_add_multiple_refinements(self):
        """Should track multiple refinements"""
        refinements = [
            ("Change 1", {'code': 'code1', 'modifications': [], 'explanation': 'Ex1'}),
            ("Change 2", {'code': 'code2', 'modifications': [], 'explanation': 'Ex2'}),
            ("Change 3", {'code': 'code3', 'modifications': [], 'explanation': 'Ex3'}),
        ]
        
        for query, result in refinements:
            self.session.add_refinement(query, result)
        
        self.assertEqual(self.session.get_version_count(), 4)  # Initial + 3
        self.assertEqual(self.session.get_current_version(), 3)
    
    def test_undo_single_step(self):
        """Should undo single refinement"""
        self.session.add_refinement("Change", {'code': 'new code', 'modifications': [], 'explanation': ''})
        
        previous_code = self.session.undo()
        
        self.assertEqual(previous_code, self.initial_code)
        self.assertEqual(self.session.get_current_version(), 0)
    
    def test_undo_multiple_steps(self):
        """Should undo multiple refinements"""
        codes = ['code1', 'code2', 'code3']
        for code in codes:
            self.session.add_refinement("Change", {'code': code, 'modifications': [], 'explanation': ''})
        
        # Undo 2 steps
        result_code = self.session.undo(2)
        
        self.assertEqual(result_code, 'code1')
        self.assertEqual(self.session.get_current_version(), 1)
    
    def test_undo_at_initial_raises_error(self):
        """Should raise error when undoing at initial version"""
        with self.assertRaises(ValueError) as cm:
            self.session.undo()
        
        self.assertIn("initial version", str(cm.exception).lower())
    
    def test_redo_single_step(self):
        """Should redo after undo"""
        self.session.add_refinement("Change", {'code': 'new code', 'modifications': [], 'explanation': ''})
        self.session.undo()
        
        next_code = self.session.redo()
        
        self.assertEqual(next_code, 'new code')
        self.assertEqual(self.session.get_current_version(), 1)
    
    def test_redo_multiple_steps(self):
        """Should redo multiple steps"""
        codes = ['code1', 'code2', 'code3']
        for code in codes:
            self.session.add_refinement("Change", {'code': code, 'modifications': [], 'explanation': ''})
        
        self.session.undo(3)  # Back to initial
        result_code = self.session.redo(2)
        
        self.assertEqual(result_code, 'code2')
        self.assertEqual(self.session.get_current_version(), 2)
    
    def test_redo_at_latest_raises_error(self):
        """Should raise error when redoing at latest version"""
        with self.assertRaises(ValueError) as cm:
            self.session.redo()
        
        self.assertIn("latest version", str(cm.exception).lower())
    
    def test_go_to_version(self):
        """Should jump to specific version"""
        for i in range(5):
            self.session.add_refinement(f"Change {i}", {'code': f'code{i}', 'modifications': [], 'explanation': ''})
        
        code = self.session.go_to_version(2)
        
        self.assertEqual(code, 'code1')  # Version 2 is the 2nd refinement (0-based: initial, code0, code1)
        self.assertEqual(self.session.get_current_version(), 2)
    
    def test_go_to_version_invalid(self):
        """Should raise error for invalid version"""
        with self.assertRaises(ValueError):
            self.session.go_to_version(10)
    
    def test_can_undo(self):
        """Should correctly report undo availability"""
        self.assertFalse(self.session.can_undo())
        
        self.session.add_refinement("Change", {'code': 'new', 'modifications': [], 'explanation': ''})
        self.assertTrue(self.session.can_undo())
    
    def test_can_redo(self):
        """Should correctly report redo availability"""
        self.assertFalse(self.session.can_redo())
        
        self.session.add_refinement("Change", {'code': 'new', 'modifications': [], 'explanation': ''})
        self.assertFalse(self.session.can_redo())
        
        self.session.undo()
        self.assertTrue(self.session.can_redo())
    
    def test_get_version_list(self):
        """Should return list of all versions"""
        self.session.add_refinement("Change 1", {'code': 'code1', 'modifications': [], 'explanation': ''})
        self.session.add_refinement("Change 2", {'code': 'code2', 'modifications': [], 'explanation': ''})
        
        versions = self.session.get_version_list()
        
        self.assertEqual(len(versions), 3)
        self.assertEqual(versions[0]['version'], 0)
        self.assertEqual(versions[0]['query'], 'Initial version')
        self.assertEqual(versions[1]['query'], 'Change 1')
        self.assertEqual(versions[2]['query'], 'Change 2')
        self.assertTrue(versions[2]['is_current'])
    
    def test_get_version_info(self):
        """Should return detailed version info"""
        self.session.add_refinement("Test", {'code': 'new', 'modifications': [{'a': 1}], 'explanation': 'test'})
        
        info = self.session.get_version_info(1)
        
        self.assertEqual(info['version'], 1)
        self.assertEqual(info['query'], 'Test')
        self.assertEqual(info['code'], 'new')
        self.assertIn('timestamp', info)
    
    def test_get_session_info(self):
        """Should return session metadata"""
        info = self.session.get_session_info()
        
        self.assertIn('session_id', info)
        self.assertIn('created_at', info)
        self.assertIn('total_versions', info)
        self.assertIn('current_version', info)
        self.assertIn('can_undo', info)
        self.assertIn('can_redo', info)
    
    def test_get_modification_summary(self):
        """Should return modification summary"""
        self.session.add_refinement("Change 1", {'code': 'new', 'modifications': [1, 2], 'explanation': ''})
        self.session.add_refinement("Change 2", {'code': 'new2', 'modifications': [3], 'explanation': ''})
        
        summary = self.session.get_modification_summary()
        
        self.assertEqual(summary['total_refinements'], 2)
        self.assertEqual(summary['total_modifications'], 3)  # 2 + 1
        self.assertEqual(len(summary['queries']), 2)
    
    def test_add_refinement_truncates_future_history(self):
        """Should truncate history when adding refinement after undo"""
        # Add 3 refinements
        for i in range(3):
            self.session.add_refinement(f"Change {i}", {'code': f'code{i}', 'modifications': [], 'explanation': ''})
        
        self.assertEqual(self.session.get_version_count(), 4)
        
        # Undo 2 steps
        self.session.undo(2)
        
        # Add new refinement
        self.session.add_refinement("New change", {'code': 'new', 'modifications': [], 'explanation': ''})
        
        # Future history should be truncated
        self.assertEqual(self.session.get_version_count(), 3)  # Initial + code0 + new
        self.assertEqual(self.session.get_current_version(), 2)
    
    def test_get_diff_between_versions(self):
        """Should generate diff between versions"""
        self.session.add_refinement("Change", {'code': 'line1\nline2\nline3', 'modifications': [], 'explanation': ''})
        
        diff = self.session.get_diff_between_versions(0, 1)
        
        self.assertIn('cylinder', diff)  # Original code
        self.assertIn('line1', diff)  # New code


class TestPipelineSessionIntegration(unittest.TestCase):
    """Test SequentialPipeline session integration"""
    
    def setUp(self):
        """Create pipeline and mocks"""
        self.mock_retriever = Mock()
        self.mock_llm_client = Mock()
        
        self.pipeline = SequentialPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            use_llm_decomposition=False
        )
    
    def test_create_refinement_session(self):
        """Should create session through pipeline"""
        code = "test code"
        session = self.pipeline.create_refinement_session(code)
        
        self.assertIsInstance(session, RefinementSession)
        self.assertEqual(session.get_current_code(), code)
    
    def test_create_session_with_custom_id(self):
        """Should create session with custom ID"""
        session = self.pipeline.create_refinement_session("code", session_id="custom-123")
        
        self.assertEqual(session.session_id, "custom-123")
    
    @patch.object(SequentialPipeline, 'process_query')
    def test_refine_in_session(self, mock_process):
        """Should refine within session"""
        # Setup
        session = self.pipeline.create_refinement_session("original code")
        
        mock_result = {
            'code': 'modified code',
            'modifications': [],
            'explanation': 'test',
            'diff': ''
        }
        mock_process.return_value = mock_result
        
        # Refine
        result = self.pipeline.refine_in_session(session, "Change it")
        
        # Verify
        mock_process.assert_called_once()
        self.assertEqual(result['code'], 'modified code')
        self.assertIn('session_version', result)
        self.assertIn('session_info', result)
        self.assertEqual(session.get_version_count(), 2)
    
    def test_get_session_code(self):
        """Should get current code from session"""
        session = self.pipeline.create_refinement_session("test code")
        
        code = self.pipeline.get_session_code(session)
        
        self.assertEqual(code, "test code")
    
    def test_undo_refinement(self):
        """Should undo through pipeline"""
        session = self.pipeline.create_refinement_session("original")
        session.add_refinement("Change", {'code': 'new', 'modifications': [], 'explanation': ''})
        
        previous_code = self.pipeline.undo_refinement(session)
        
        self.assertEqual(previous_code, "original")
    
    def test_undo_refinement_raises_error(self):
        """Should raise error when cannot undo"""
        session = self.pipeline.create_refinement_session("original")
        
        with self.assertRaises(ValueError):
            self.pipeline.undo_refinement(session)
    
    def test_redo_refinement(self):
        """Should redo through pipeline"""
        session = self.pipeline.create_refinement_session("original")
        session.add_refinement("Change", {'code': 'new', 'modifications': [], 'explanation': ''})
        session.undo()
        
        next_code = self.pipeline.redo_refinement(session)
        
        self.assertEqual(next_code, "new")
    
    def test_redo_refinement_raises_error(self):
        """Should raise error when cannot redo"""
        session = self.pipeline.create_refinement_session("original")
        
        with self.assertRaises(ValueError):
            self.pipeline.redo_refinement(session)
    
    def test_get_session_versions(self):
        """Should get version list through pipeline"""
        session = self.pipeline.create_refinement_session("original")
        session.add_refinement("Change 1", {'code': 'new1', 'modifications': [], 'explanation': ''})
        session.add_refinement("Change 2", {'code': 'new2', 'modifications': [], 'explanation': ''})
        
        versions = self.pipeline.get_session_versions(session)
        
        self.assertEqual(len(versions), 3)
        self.assertEqual(versions[1]['query'], 'Change 1')


class TestRefinementVersion(unittest.TestCase):
    """Test RefinementVersion dataclass"""
    
    def test_version_to_dict(self):
        """Should convert version to dictionary"""
        from datetime import datetime
        
        version = RefinementVersion(
            version=1,
            timestamp=datetime.now(),
            code='test code' * 50,  # Long code
            query='Test query',
            modifications=[],
            explanation='Test',
            diff='test diff'
        )
        
        result = version.to_dict()
        
        self.assertEqual(result['version'], 1)
        self.assertEqual(result['query'], 'Test query')
        self.assertIn('timestamp', result)
        self.assertIn('code_preview', result)
        # Preview should be truncated
        self.assertLess(len(result['code_preview']), len(version.code))


if __name__ == '__main__':
    unittest.main(verbosity=2)
