#!/usr/bin/env python3
"""Tests for JSON Logger"""

import sys
from pathlib import Path
import unittest
import json
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-generation'))

from json_logger import PipelineLogger


class TestJSONLogger(unittest.TestCase):
    """Test JSON logger functionality"""
    
    def setUp(self):
        """Create temporary directory for tests"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.logger = PipelineLogger(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_log_query(self):
        """Test logging query"""
        self.logger.log_query("Test query")
        self.assertEqual(self.logger.current_log["query"], "Test query")
        self.assertIsNotNone(self.logger.current_log["timestamp"])
    
    def test_log_decomposition(self):
        """Test logging decomposition"""
        input_data = {"query": "test"}
        output_data = {"steps": []}
        
        self.logger.log_decomposition(input_data, output_data)
        
        self.assertIsNotNone(self.logger.current_log["decomposition"])
        self.assertEqual(self.logger.current_log["decomposition"]["input"], input_data)
        self.assertEqual(self.logger.current_log["decomposition"]["output"], output_data)
    
    def test_log_step(self):
        """Test logging generation step"""
        input_data = {"step": 1}
        output_data = {"code": "test"}
        
        self.logger.log_step(1, input_data, output_data)
        
        self.assertEqual(len(self.logger.current_log["steps"]), 1)
        self.assertEqual(self.logger.current_log["steps"][0]["step_number"], 1)
    
    def test_log_validation(self):
        """Test logging validation"""
        input_data = {"errors": []}
        output_data = {"fixed": True}
        
        self.logger.log_validation(input_data, output_data)
        
        self.assertIsNotNone(self.logger.current_log["validation"])
        self.assertEqual(len(self.logger.current_log["validation"]), 1)
    
    def test_save_and_load(self):
        """Test saving and loading logs"""
        self.logger.log_query("Test save")
        self.logger.log_decomposition({"test": "input"}, {"test": "output"})
        
        # Save
        filepath = self.logger.save("test_log.json")
        self.assertTrue(filepath.exists())
        
        # Load
        loaded = PipelineLogger.load_log(filepath)
        self.assertEqual(loaded["query"], "Test save")
        self.assertIn("decomposition", loaded)
    
    def test_clear(self):
        """Test clearing log"""
        self.logger.log_query("Test")
        self.logger.clear()
        
        self.assertIsNone(self.logger.current_log["query"])
        self.assertEqual(len(self.logger.current_log["steps"]), 0)
    
    def test_format_summary(self):
        """Test formatting log summary"""
        log = {
            "timestamp": "2025-10-22",
            "query": "Test query",
            "decomposition": {
                "output": {
                    "understanding": "Test understanding",
                    "steps": [{"step": 1}, {"step": 2}]
                }
            },
            "steps": [
                {"step_number": 1, "output": {"code": "test code"}},
                {"step_number": 2, "output": {"code": "more code"}}
            ],
            "final_result": {
                "final_code": {"complete": "final code here"}
            }
        }
        
        summary = PipelineLogger.format_log_summary(log)
        
        self.assertIn("PIPELINE LOG SUMMARY", summary)
        self.assertIn("Test query", summary)
        self.assertIn("2 steps", summary)
        self.assertIn("Generation Steps: 2", summary)


if __name__ == '__main__':
    unittest.main()
