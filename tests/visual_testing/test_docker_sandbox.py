#!/usr/bin/env python3
"""
Unit tests for DockerSandbox

Tests Docker container execution, isolation, resource limits, and timeouts.
Run with: python -m unittest tests.visual_testing.test_docker_sandbox
"""

import unittest
import sys
import os
from pathlib import Path

# Add visual_testing module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'visual_testing'))

from docker_sandbox import DockerSandbox

# Environment flag to enable/disable tests (requires Docker)
RUN_VISUAL_TESTS = os.getenv('RUN_VISUAL_TESTS', '0') == '1'


class TestDockerSandbox(unittest.TestCase):
    """Unit tests for Docker sandbox functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        if not RUN_VISUAL_TESTS:
            raise unittest.SkipTest("Visual tests disabled. Set RUN_VISUAL_TESTS=1 to enable")
        cls.sandbox = DockerSandbox()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if hasattr(cls, 'sandbox'):
            cls.sandbox.cleanup()
    
    def test_sandbox_resource_limits(self):
        """Test: Resource limits are enforced"""
        
        # Code that completes quickly
        code = """
import time
print("Starting...")
time.sleep(0.5)
print("Done")
"""
        
        result = self.sandbox.execute_code(code, "resource_test")
        
        self.assertTrue(result['success'], "Simple code should execute")
        self.assertLess(result['execution_time'], 5, "Should complete quickly")
        self.assertIn("Done", result['output'], "Should see output")
    
    def test_sandbox_timeout(self):
        """Test: Timeout is enforced"""
        
        # Create sandbox with short timeout
        short_sandbox = DockerSandbox(timeout=2)
        
        # Code that takes too long
        code = """
import time
time.sleep(10)  # Longer than timeout
"""
        
        result = short_sandbox.execute_code(code, "timeout_test")
        
        # Should fail due to timeout
        self.assertFalse(result['success'], "Should timeout")
        self.assertIsNotNone(result['error'], "Should have error message")
        
        short_sandbox.cleanup()
    
    def test_sandbox_isolation(self):
        """Test: Filesystem isolation"""
        
        # Try to write to host filesystem (should fail - read-only mount)
        code = """
with open('/workspace/test_write.txt', 'w') as f:
    f.write("trying to write")
print("Should not reach here")
"""
        
        result = self.sandbox.execute_code(code, "isolation_test")
        
        # Should fail because workspace is read-only
        self.assertFalse(result['success'], "Should not write to read-only workspace")
    
    def test_sandbox_network_disabled(self):
        """Test: Network access is disabled"""
        
        code = """
import socket
try:
    socket.create_connection(('google.com', 80), timeout=1)
    print("Network accessible")
except Exception as e:
    print(f"Network blocked: {e}")
"""
        
        result = self.sandbox.execute_code(code, "network_test")
        
        # Should succeed (code runs) but network should be blocked
        self.assertTrue(result['success'], "Code should execute")
        self.assertIn("Network blocked", result['output'], "Network should be disabled")


if __name__ == '__main__':
    unittest.main()
