#!/usr/bin/env python3
"""
VTK Code Validator

Post-generation validation layer using VTK API index.
Validates imports, classes, and method usage.
"""

import ast
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

from vtk_api_server import VTKAPIIndex


@dataclass
class ValidationError:
    """Single validation error"""
    error_type: str  # 'import', 'unknown_class', 'unknown_method'
    message: str
    line: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: List[ValidationError]
    code: str
    
    @property
    def has_errors(self):
        return len(self.errors) > 0
    
    def format_errors(self) -> str:
        """Format errors for LLM correction prompt"""
        if not self.errors:
            return "No errors found."
        
        formatted = []
        for i, error in enumerate(self.errors, 1):
            formatted.append(f"{i}. {error.error_type.upper()}: {error.message}")
            if error.line:
                formatted.append(f"   Line: {error.line}")
            if error.suggestion:
                formatted.append(f"   Suggestion: {error.suggestion}")
        
        return "\n".join(formatted)


class VTKCodeValidator:
    """Validate generated VTK code using API index"""
    
    def __init__(self, api_index: VTKAPIIndex):
        """
        Initialize validator
        
        Args:
            api_index: Loaded VTK API index
        """
        self.api = api_index
    
    def validate_code(self, code: str) -> ValidationResult:
        """
        Validate imports, classes, and methods in generated code
        
        Args:
            code: Python code to validate
            
        Returns:
            ValidationResult with any errors found
        """
        errors = []
        
        # 1. Validate imports
        import_errors = self._validate_imports(code)
        errors.extend(import_errors)
        
        # 2. Validate class usage
        class_errors = self._validate_classes(code)
        errors.extend(class_errors)
        
        # 3. Validate method usage (simple check for obvious hallucinations)
        method_errors = self._validate_methods(code)
        errors.extend(method_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            code=code
        )
    
    def _validate_imports(self, code: str) -> List[ValidationError]:
        """Validate all VTK import statements"""
        errors = []
        imports = self._extract_imports(code)
        
        for imp in imports:
            # Only validate VTK imports
            if 'vtk' not in imp.lower():
                continue
            
            # Pass full code context for smart validation
            result = self.api.validate_import(imp, code_context=code)
            if not result['valid']:
                errors.append(ValidationError(
                    error_type='import',
                    message=result['message'],
                    line=imp,
                    suggestion=result.get('suggested')
                ))
        
        return errors
    
    def _validate_classes(self, code: str) -> List[ValidationError]:
        """Validate VTK class usage"""
        errors = []
        classes = self._extract_class_instantiations(code)
        
        for cls in classes:
            # Only validate VTK classes
            if not cls.startswith('vtk'):
                continue
            
            info = self.api.get_class_info(cls)
            if not info:
                suggestion = self._suggest_similar_class(cls)
                message = (
                    f"INVALID: Class '{cls}' not found in VTK API.\n"
                    f"  This is likely a hallucination or typo.\n\n"
                    f"  SMALLEST CHANGE: Replace '{cls}' with a valid VTK class name"
                )
                if suggestion:
                    message += f" (try: {suggestion})"
                
                errors.append(ValidationError(
                    error_type='unknown_class',
                    message=message,
                    line=None,
                    suggestion=suggestion
                ))
        
        return errors
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract all import statements from code"""
        imports = []
        lines = code.split('\n')
        
        in_multiline = False
        current_import = []
        
        for line in lines:
            stripped = line.strip()
            
            # Start of import
            if stripped.startswith('import ') or stripped.startswith('from '):
                current_import = [line]
                
                if '(' in line and ')' not in line:
                    in_multiline = True
                else:
                    imports.append(line)
                    current_import = []
            
            # Continuation
            elif in_multiline:
                current_import.append(line)
                if ')' in line:
                    in_multiline = False
                    imports.append('\n'.join(current_import))
                    current_import = []
        
        return imports
    
    def _extract_class_instantiations(self, code: str) -> List[str]:
        """Extract VTK class instantiations (e.g., vtkPolyDataMapper())"""
        classes = set()
        
        # Pattern: vtkClassName()
        pattern = r'\b(vtk[A-Z][a-zA-Z0-9]*)\s*\('
        matches = re.findall(pattern, code)
        classes.update(matches)
        
        return list(classes)
    
    def _validate_methods(self, code: str) -> List[ValidationError]:
        """
        Validate VTK method calls using type tracking
        
        Strategy:
        1. Track variable types from instantiations (e.g., mapper = vtkPolyDataMapper())
        2. Validate method calls against the tracked type (e.g., mapper.SetInputData())
        3. Use MCP's get_method_info(class_name, method_name) to check validity
        """
        errors = []
        
        # Step 1: Track variable types from instantiations
        var_types = self._track_variable_types(code)
        
        # Step 2: Extract method calls with object references
        method_calls = self._extract_method_calls_with_objects(code)
        
        for obj_name, method_name, line in method_calls:
            # Get the type of the object
            class_name = var_types.get(obj_name)
            if not class_name:
                # Can't determine type, skip validation
                continue
            
            # Check if this method exists for this class
            method_info = self.api.get_method_info(class_name, method_name)
            if not method_info:
                # Find similar methods from the same class
                suggestion = self._suggest_similar_method_from_class(class_name, method_name)
                
                # Create strong, actionable error message
                if suggestion:
                    message = (
                        f"method: INVALID: Method '{method_name}()' doesn't exist on '{class_name}'.\n"
                        f"  REPLACE THIS EXACT LINE:\n"
                        f"    {line.strip()}\n"
                        f"  WITH:\n"
                        f"    {line.strip().replace(method_name, suggestion)}\n\n"
                        f"  REQUIRED: Change '{method_name}' to '{suggestion}' - this is the correct method name."
                    )
                else:
                    message = (
                        f"method: INVALID: Method '{method_name}()' doesn't exist on '{class_name}'.\n"
                        f"  REMOVE THIS LINE:\n"
                        f"    {line.strip()}\n\n"
                        f"  REQUIRED: Delete this line completely - this method does not exist in VTK."
                    )
                
                errors.append(ValidationError(
                    error_type='method',
                    message=message,
                    line=line.strip() if line else None,
                    suggestion=suggestion
                ))
        
        return errors
    
    def _track_variable_types(self, code: str) -> Dict[str, str]:
        """
        Track variable types from VTK class instantiations
        
        Examples:
            mapper = vtkPolyDataMapper() → {'mapper': 'vtkPolyDataMapper'}
            actor = vtk.vtkActor() → {'actor': 'vtkActor'}
        
        Returns:
            Dict mapping variable names to VTK class names
        """
        var_types = {}
        lines = code.split('\n')
        
        # Pattern 1: var = vtkClassName()
        pattern1 = r'(\w+)\s*=\s*(vtk[A-Z][a-zA-Z0-9]*)\s*\('
        
        # Pattern 2: var = vtk.vtkClassName()
        pattern2 = r'(\w+)\s*=\s*vtk\.(vtk[A-Z][a-zA-Z0-9]*)\s*\('
        
        for line in lines:
            # Try pattern 1
            matches = re.findall(pattern1, line)
            for var_name, class_name in matches:
                var_types[var_name] = class_name
            
            # Try pattern 2
            matches = re.findall(pattern2, line)
            for var_name, class_name in matches:
                var_types[var_name] = class_name
        
        return var_types
    
    def _extract_method_calls_with_objects(self, code: str) -> List[tuple]:
        """
        Extract method calls with object references
        
        Returns:
            List of (obj_name, method_name, line) tuples
        """
        method_calls = []
        lines = code.split('\n')
        
        # Pattern: obj_name.MethodName(
        pattern = r'(\w+)\.([A-Z][a-zA-Z0-9_]*)\s*\('
        
        for line in lines:
            matches = re.findall(pattern, line)
            for obj_name, method_name in matches:
                method_calls.append((obj_name, method_name, line))
        
        return method_calls
    
    def _extract_method_calls(self, code: str) -> List[tuple]:
        """
        Extract method calls from code (deprecated - use _extract_method_calls_with_objects)
        
        Returns:
            List of (method_name, line) tuples
        """
        method_calls = []
        lines = code.split('\n')
        
        # Pattern: obj.method_name(
        pattern = r'\.([A-Z][a-zA-Z0-9_]*)\s*\('
        
        for line in lines:
            matches = re.findall(pattern, line)
            for method in matches:
                method_calls.append((method, line))
        
        return method_calls
    
    
    def _suggest_similar_method(self, method_name: str) -> Optional[str]:
        """Suggest similar method names"""
        # Common typos and deprecated methods
        common_fixes = {
            'SetColorMap': 'SetLookupTable',
            'AddInput': 'AddInputData',
            'SetInput': 'SetInputData',
            'GetInput': 'GetInputDataObject',
        }
        
        if method_name in common_fixes:
            return f"Did you mean: {common_fixes[method_name]}()?"
        
        # Try fuzzy search
        results = self.api.search_classes(method_name[:8], limit=3)
        if results:
            return f"Check spelling or see similar: {', '.join(r['class_name'] for r in results[:2])}"
        
        return None
    
    def _suggest_similar_method_from_class(self, class_name: str, method_name: str) -> Optional[str]:
        """
        Suggest similar method names from the actual class
        
        Args:
            class_name: VTK class name (e.g., 'vtkExodusIIReader')
            method_name: Invalid method name to find alternatives for
            
        Returns:
            Suggested method name or None
        """
        # Get all valid methods from the class using MCP's get_method_info approach
        class_info = self.api.get_class_info(class_name)
        if not class_info:
            return None
        
        # Extract methods from structured_docs
        valid_methods = []
        structured_docs = class_info.get('structured_docs', {})
        if structured_docs:
            sections = structured_docs.get('sections', {})
            for section_data in sections.values():
                if 'methods' in section_data:
                    valid_methods.extend(section_data['methods'].keys())
        
        if not valid_methods:
            return None
        
        # First try exact match (case-insensitive)
        for valid_method in valid_methods:
            if valid_method.lower() == method_name.lower():
                return valid_method
        
        # Try fuzzy match using difflib
        import difflib
        close_matches = difflib.get_close_matches(method_name, valid_methods, n=1, cutoff=0.6)
        if close_matches:
            return close_matches[0]
        
        # Try finding methods with similar prefix
        method_prefix = method_name[:4]  # First 4 chars
        similar_prefix = [m for m in valid_methods if m.startswith(method_prefix)]
        if similar_prefix:
            return similar_prefix[0]
        
        return None
    
    def _suggest_similar_class(self, class_name: str) -> Optional[str]:
        """Suggest similar class names for typos"""
        # Simple fuzzy search
        results = self.api.search_classes(class_name[:10], limit=3)
        
        if results:
            suggestions = [r['class_name'] for r in results]
            return f"Did you mean: {', '.join(suggestions)}?"
        
        return None


def load_validator(api_docs_path: Path = None) -> VTKCodeValidator:
    """
    Convenience function to load validator
    
    Args:
        api_docs_path: Path to vtk-python-docs.jsonl (raw format)
                      (defaults to ../data/raw/vtk-python-docs.jsonl)
    
    Returns:
        Initialized VTKCodeValidator
    """
    if api_docs_path is None:
        api_docs_path = Path(__file__).parent.parent / "data" / "raw" / "vtk-python-docs.jsonl"
    
    api_index = VTKAPIIndex(api_docs_path)
    return VTKCodeValidator(api_index)


# Example usage
if __name__ == "__main__":
    # Test the validator
    validator = load_validator()
    
    test_code = """
from vtkmodules.vtkCommonDataModel import vtkPolyDataMapper
from vtkmodules.vtkRenderingCore import vtkActor

def main():
    mapper = vtkPolyDataMapper()
    actor = vtkActor()
    reader = vtkSTLReaderr()  # Typo
"""
    
    result = validator.validate_code(test_code)
    
    print("Validation Result:")
    print(f"Valid: {result.is_valid}")
    print(f"\nErrors ({len(result.errors)}):")
    print(result.format_errors())
