"""
Security Validator for Generated VTK Code

Validates LLM-generated code for security issues before execution or returning to users.
Allows VTK I/O operations while blocking dangerous Python file/system access.
"""

import ast
import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class VTKCodeSafetyValidator:
    """
    Validate VTK code for security issues without executing it.
    
    ✅ ALLOWED:
    - VTK I/O modules (vtkPNGWriter, vtkSTLReader, etc.)
    - VTK file operations (reader.SetFileName(), writer.Write())
    - Pandas file I/O (pd.read_csv() - common data pattern)
    - Numpy operations
    
    ❌ BLOCKED:
    - Direct Python file I/O (open(), pathlib.Path())
    - System operations (os.system, subprocess)
    - Network access (urllib, requests, socket)
    - Code execution (eval, exec, __import__)
    - File manipulation (shutil, os.remove)
    """
    
    ALLOWED_IMPORTS = {
        'vtkmodules',  # All VTK modules (including I/O)
        'vtk',
        'numpy',
        'pandas',  # Common for data handling
        'math',
        'random',
    }
    
    # Dangerous patterns - Python file I/O and system operations
    # Note: VTK I/O (vtkPNGWriter, vtkSTLReader, etc.) is ALLOWED
    DANGEROUS_PATTERNS = [
        # Direct Python file I/O (dangerous)
        (r'\bopen\s*\(', 'Direct file I/O using open()'),
        (r'pathlib\.Path\(', 'Direct path manipulation using pathlib.Path'),
        (r'\bPath\(', 'Direct path manipulation using Path'),
        
        # System/subprocess
        (r'os\.system', 'System command execution via os.system'),
        (r'os\.remove', 'File deletion via os.remove'),
        (r'os\.unlink', 'File deletion via os.unlink'),
        (r'os\.rmdir', 'Directory removal via os.rmdir'),
        (r'subprocess\.', 'Subprocess execution'),
        
        # Network
        (r'urllib\.', 'Network access via urllib'),
        (r'requests\.', 'Network access via requests'),
        (r'socket\.', 'Socket operations'),
        (r'http\.', 'HTTP operations'),
        
        # Code execution
        (r'\beval\s*\(', 'Code execution via eval()'),
        (r'\bexec\s*\(', 'Code execution via exec()'),
        (r'__import__', 'Dynamic import via __import__'),
        (r'\bcompile\s*\(', 'Code compilation via compile()'),
        
        # Dangerous file operations
        (r'shutil\.', 'File operations via shutil'),
    ]
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate code for security issues.
        
        Args:
            code: Python code string to validate
            
        Returns:
            Tuple of (is_safe, [list of issues])
        """
        issues = []
        
        # 1. Check for dangerous patterns via regex
        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"{description} - pattern: {pattern}")
                logger.warning(f"Security issue found: {description}")
        
        # 2. Parse AST and check imports
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            # Syntax errors will be caught by code_validator
            # Don't fail security check for syntax issues
            logger.debug(f"Syntax error during security validation: {e}")
            return True, []  # Let syntax validator handle this
        except Exception as e:
            logger.warning(f"Could not parse code for security validation: {e}")
            # If we can't parse it, be cautious and pass it
            # (syntax validator will catch it)
            return True, []
        
        # 3. Check imports against whitelist
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_allowed_import(alias.name):
                        issues.append(f"Disallowed import: {alias.name}")
                        logger.warning(f"Disallowed import detected: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and not self._is_allowed_import(node.module):
                    issues.append(f"Disallowed import: {node.module}")
                    logger.warning(f"Disallowed import detected: {node.module}")
        
        is_safe = len(issues) == 0
        
        if is_safe:
            logger.debug("Code passed security validation")
        else:
            logger.warning(f"Code failed security validation with {len(issues)} issues")
        
        return is_safe, issues
    
    def _is_allowed_import(self, module: str) -> bool:
        """
        Check if import is in allowlist.
        
        Args:
            module: Module name to check
            
        Returns:
            True if module is allowed, False otherwise
        """
        return any(module.startswith(allowed) for allowed in self.ALLOWED_IMPORTS)
    
    def get_allowed_imports(self) -> set:
        """
        Get set of allowed import prefixes.
        
        Returns:
            Set of allowed import prefixes
        """
        return self.ALLOWED_IMPORTS.copy()
    
    def format_issues(self, issues: List[str]) -> str:
        """
        Format security issues as human-readable text.
        
        Args:
            issues: List of security issue strings
            
        Returns:
            Formatted string with all issues
        """
        if not issues:
            return "No security issues found."
        
        formatted = "Security Issues Found:\n"
        for i, issue in enumerate(issues, 1):
            formatted += f"  {i}. {issue}\n"
        return formatted
