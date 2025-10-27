"""
Code Validator with JSON-based LLM communication

Validates generated Python code and attempts to fix issues using structured JSON.
"""

import ast
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a code validation error"""
    error_type: str
    message: str
    line: Optional[int] = None
    suggestion: Optional[str] = None


class CodeValidator:
    """Validates Python code using AST parsing"""
    
    def __init__(self):
        """Initialize validator"""
        pass
    
    def validate_code(self, code: str) -> Tuple[bool, List[ValidationError]]:
        """
        Validate Python code for syntax and common issues
        
        Args:
            code: Python code string to validate
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check if code is empty
        if not code or not code.strip():
            errors.append(ValidationError(
                error_type="empty_code",
                message="Code is empty",
                line=None
            ))
            return False, errors
        
        # Syntax check using AST
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(ValidationError(
                error_type="syntax_error",
                message=str(e.msg),
                line=e.lineno,
                suggestion="Fix syntax error"
            ))
            return False, errors
        except Exception as e:
            errors.append(ValidationError(
                error_type="parse_error",
                message=str(e),
                line=None
            ))
            return False, errors
        
        # Additional semantic checks
        try:
            tree = ast.parse(code)
            
            # Check for common issues
            has_statements = len(tree.body) > 0
            if not has_statements:
                errors.append(ValidationError(
                    error_type="no_statements",
                    message="Code has no executable statements",
                    line=None
                ))
                return False, errors
            
        except Exception as e:
            logger.warning(f"Semantic validation failed: {e}")
        
        # If we got here, code is valid
        return True, errors
    
    def format_errors(self, errors: List[ValidationError]) -> str:
        """
        Format validation errors as human-readable text
        
        Args:
            errors: List of ValidationError objects
            
        Returns:
            Formatted error string
        """
        if not errors:
            return "No errors"
        
        lines = []
        for i, error in enumerate(errors, 1):
            location = f" (line {error.line})" if error.line else ""
            lines.append(f"{i}. {error.error_type}{location}: {error.message}")
            if error.suggestion:
                lines.append(f"   Suggestion: {error.suggestion}")
        
        return "\n".join(lines)


class LLMCodeValidator:
    """Uses LLM with JSON communication to validate and fix code"""
    
    def __init__(self, llm_client):
        """
        Initialize LLM-based validator
        
        Args:
            llm_client: LLMClient instance for JSON generation
        """
        self.llm_client = llm_client
        self.basic_validator = CodeValidator()
    
    def validate_and_fix(
        self,
        code: str,
        context: str,
        max_retries: int = 2
    ) -> Tuple[str, List[Dict], bool]:
        """
        Validate code and attempt to fix issues using LLM
        
        Args:
            code: Code to validate
            context: Context about what the code should do
            max_retries: Maximum number of fix attempts
            
        Returns:
            Tuple of (fixed_code, changes_made, success)
        """
        # First, do basic validation
        is_valid, errors = self.basic_validator.validate_code(code)
        
        if is_valid:
            pass  # Validation passed
            return code, [], True
        
        # Code has errors - attempt to fix
        logger.warning(f"⚠️  Code validation failed: {len(errors)} error(s)")
        logger.debug(f"Errors:\n{self.basic_validator.format_errors(errors)}")
        
        current_code = code
        all_changes = []
        
        for attempt in range(1, max_retries + 1):
            pass  # Attempting fix
            
            # Build validation input
            validation_input = {
                "code": current_code,
                "context": context,
                "errors": [
                    {
                        "error_type": e.error_type,
                        "message": e.message,
                        "line": e.line
                    }
                    for e in errors
                ],
                "instructions": {
                    "task": "Fix the code errors while preserving functionality",
                    "requirements": [
                        "Fix all syntax errors",
                        "Maintain the original logic and purpose",
                        "Do not change working parts of the code",
                        "Return complete corrected code",
                        "Document each change made"
                    ]
                }
            }
            
            try:
                # Call LLM to fix code
                result = self.llm_client.generate_json(
                    prompt_data=validation_input,
                    schema_name="ValidationOutput",
                    temperature=0.1
                )
                
                fixed_code = result.get('fixed_code', '')
                changes = result.get('changes_made', [])
                
                # Validate the fixed code
                is_valid, new_errors = self.basic_validator.validate_code(fixed_code)
                
                if is_valid:
                    pass  # Code fixed successfully
                    all_changes.extend(changes)
                    return fixed_code, all_changes, True
                
                # Still has errors, update for next iteration
                current_code = fixed_code
                errors = new_errors
                all_changes.extend(changes)
                
            except Exception as e:
                logger.error(f"Error during validation attempt {attempt}: {e}")
                break
        
        # Max retries exhausted
        logger.warning(f"⚠️  Could not fix code after {max_retries} attempts")
        return current_code, all_changes, False
