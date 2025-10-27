#!/usr/bin/env python3
"""
Sequential Pipeline for Complex VTK Queries

Breaks complex queries into steps using LLM decomposition,
retrieves targeted chunks per step, and generates solutions incrementally.

Key Benefits:
- LLM-powered intelligent query decomposition
- Step-by-step generation with validation
- Targeted retrieval per substep
- Quality-boosted retrieval (pythonic+modular examples preferred)
- Handles multi-part VTK workflows

Quality Boosting:
- Pythonic API examples: +20% score boost
- Modular imports: +15% score boost
- Combined boost: +35% for ideal examples (971 gold-standard examples)

Production Usage (LLM Decomposition):
-------------------------------------
```python
from sequential_pipeline import SequentialPipeline

# Uses LLM for intelligent decomposition (production mode)
pipeline = SequentialPipeline(
    llm_client=None,  # Creates from .env if None
    use_llm_decomposition=True  # Default
)

# Process complex query - LLM will decompose it
result = pipeline.process_query(
    "Read a CSV file, edit values, and visualize as 3D points"
)
```

Notes:
------
- LLM decomposition is REQUIRED (no fallbacks)
- If decomposition fails, an error is raised
- MCP mode is deprecated (Cascade only, not for production)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
import re
import json
import logging

logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / 'retrieval-pipeline'))
sys.path.append(str(Path(__file__).parent.parent / 'grounding-prompting'))
sys.path.append(str(Path(__file__).parent.parent / 'post-processing'))
sys.path.append(str(Path(__file__).parent.parent / 'api-mcp'))

from task_specific_retriever import TaskSpecificRetriever, TaskType
from llm_client import LLMClient
from generator import VTKRAGGenerator
from code_validator import LLMCodeValidator
from refinement_session import RefinementSession
from security_validator import VTKCodeSafetyValidator

# API validation imports
try:
    from vtk_validator import load_validator
    API_VALIDATION_AVAILABLE = True
except ImportError:
    API_VALIDATION_AVAILABLE = False


@dataclass
class QueryStep:
    """Single step in query decomposition"""
    step_number: int
    description: str
    query: str
    focus: str  # What this step accomplishes


@dataclass
class StepResult:
    """Result for a single step"""
    step: QueryStep
    retrieved_chunks: List[Any]
    token_count: int
    solution: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete pipeline result"""
    query: str
    steps: List[QueryStep]
    code: str
    explanation: str
    chunk_ids_used: List[str]
    step_solutions: List[Dict]
    validation_attempted: bool = False
    validation_errors_found: int = 0
    validation_retries: int = 0
    validation_final_status: str = "not_run"
    api_validation_attempted: bool = False
    api_validation_passed: bool = True
    api_validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.api_validation_errors is None:
            self.api_validation_errors = []


class SequentialPipeline:
    """
    Sequential thinking pipeline for complex VTK queries
    
    Decomposes complex queries into steps, retrieves per step,
    and generates solutions incrementally.
    """
    
    def __init__(
        self,
        retriever: Optional[TaskSpecificRetriever] = None,
        qdrant_url: str = "http://localhost:6333",
        llm_client: Optional[LLMClient] = None,
        use_llm_decomposition: bool = True,
        enable_validation: bool = True,
        enable_api_validation: bool = True,
        validation_max_retries: int = 2
    ):
        """
        Initialize sequential pipeline
        
        Args:
            retriever: TaskSpecificRetriever instance (creates one if None)
            qdrant_url: Qdrant server URL
            llm_client: LLM client for decomposition (creates from env if None)
            use_llm_decomposition: Use LLM for decomposition (True=LLM, False=heuristic)
            enable_validation: Enable LLM code validation
            enable_api_validation: Enable VTK API validation (catches hallucinations)
            validation_max_retries: Max validation retries
        """
        self.retriever = retriever or TaskSpecificRetriever(qdrant_url=qdrant_url)
        self.llm_client = llm_client or (LLMClient() if use_llm_decomposition else None)
        self.use_llm_decomposition = use_llm_decomposition and self.llm_client is not None
        self.enable_validation = enable_validation
        self.enable_api_validation = enable_api_validation
        self.validation_max_retries = validation_max_retries
        
        # Initialize generation components
        self.generator = VTKRAGGenerator(
            validate_code=False,  # We handle validation in JSON pipeline now
            validation_max_retries=validation_max_retries
        )
        # self.parser = ResponseParser()  # DEPRECATED - all responses now JSON
        
        # Initialize LLM validator if enabled
        self.validator = LLMCodeValidator(self.llm_client) if enable_validation and self.llm_client else None
        
        # Initialize API validator if enabled and available
        self.api_validator = None
        if enable_api_validation and API_VALIDATION_AVAILABLE:
            try:
                self.api_validator = load_validator()
                logger.info("API validator loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load API validator: {e}")
        elif enable_api_validation and not API_VALIDATION_AVAILABLE:
            logger.warning("API validation requested but vtk_validator not available")
        
        # Initialize security validator (always enabled)
        self.security_validator = VTKCodeSafetyValidator()
    
    def decompose_query_llm(self, query: str) -> List[QueryStep]:
        """
        Decompose query using LLM with structured JSON communication
        
        Args:
            query: Complex query to decompose
            
        Returns:
            List of QueryStep objects
        """
        if not self.use_llm_decomposition:
            return self.decompose_query_heuristic(query)
        
        # Build structured decomposition input using centralized prompts
        from prompt_templates import VTKPromptTemplate
        prompt_template = VTKPromptTemplate()
        
        decomposition_input = {
            "query": query,
            "instructions": prompt_template.get_decomposition_instructions()
        }
        
        try:
            # Call LLM with structured JSON input/output
            result = self.llm_client.generate_json(
                prompt_data=decomposition_input,
                schema_name="DecompositionOutput",
                temperature=0.3
            )
            
            # Convert to QueryStep objects
            steps = []
            for step_data in result.get('steps', []):
                steps.append(QueryStep(
                    step_number=step_data['step_number'],
                    description=step_data['description'],
                    query=step_data.get('search_query', step_data['description']),
                    focus=step_data.get('focus', 'general')
                ))
            
            if not steps:
                raise ValueError("LLM returned empty steps")
            
            return steps
                
        except Exception as e:
            raise ValueError(f"LLM decomposition failed: {e}")
    
    def decompose_query(self, query: str, use_llm: Optional[bool] = None) -> List[QueryStep]:
        """
        Decompose query into sequential steps
        
        Args:
            query: Query to decompose
            use_llm: Override for decomposition method
                    - None: Use instance default (use_llm_decomposition)
                    - True: Force LLM decomposition
                    - False: Force heuristic decomposition (deterministic)
        
        Returns:
            List of QueryStep objects
        """
        # Determine which method to use
        should_use_llm = use_llm if use_llm is not None else self.use_llm_decomposition
        
        if should_use_llm:
            return self.decompose_query_llm(query)
        else:
            return self.decompose_query_heuristic(query)
    
    def decompose_query_heuristic(self, query: str) -> List[QueryStep]:
        """
        Decompose query using vocabulary-based heuristic with meaningful steps
        
        Creates specialized sub-queries for each detected operation in the VTK pipeline.
        """
        q = query.lower()
        steps = []
        step_num = 1
        
        # Step 1: Data I/O (read/load)
        if any(t in q for t in ['read', 'load', 'import', 'open']):
            # Extract file type if present
            file_types = ['stl', 'obj', 'vtk', 'vtp', 'vtu', 'ply', 'csv', 'xml', 'dicom', 
                         'hdr', 'slc', 'plot3d', 'exodus', '3ds', 'png', 'jpg', 'jpeg']
            file_type = next((ft for ft in file_types if ft in q), 'file')
            
            steps.append(QueryStep(
                step_number=step_num,
                description=f"Read/load {file_type} data",
                query=f"VTK read {file_type} file data",
                focus="data_io"
            ))
            step_num += 1
        
        # Step 2: Create/generate (if no read operation)
        elif any(t in q for t in ['create', 'generate', 'build', 'construct']):
            # Extract what to create
            objects = ['cylinder', 'sphere', 'cone', 'cube', 'plane', 'arrow', 'polygon',
                      'polyhedron', 'line', 'point', 'cell', 'grid', 'surface']
            obj_type = next((obj for obj in objects if obj in q), 'geometry')
            
            steps.append(QueryStep(
                step_number=step_num,
                description=f"Create {obj_type}",
                query=f"VTK create {obj_type}",
                focus="data_creation"
            ))
            step_num += 1
        
        # Step 3: Processing/filtering operations
        filters = {
            'smooth': ('smooth', 'smoothing filter'),
            'threshold': ('threshold', 'threshold filter'),
            'clip': ('clip', 'clipping'),
            'slice': ('slice', 'slicing'),
            'contour': ('contour', 'contouring'),
            'isosurface': ('isosurface', 'isosurface extraction'),
            'decimate': ('decimate', 'decimation'),
            'warp': ('warp', 'warping'),
            'edit': ('edit', 'editing'),
            'modify': ('modify', 'modification'),
            'process': ('process', 'processing'),
            'convert': ('convert', 'conversion'),
            'filter': ('filter', 'filtering')
        }
        
        for keyword, (term, description) in filters.items():
            if keyword in q:
                steps.append(QueryStep(
                    step_number=step_num,
                    description=f"Apply {description}",
                    query=f"VTK {term} filter",
                    focus="filtering"
                ))
                step_num += 1
        
        # Step 4: Transformations
        transforms = ['rotate', 'scale', 'translate', 'move', 'position', 'transform']
        if any(t in q for t in transforms):
            trans_type = next((t for t in transforms if t in q), 'transform')
            steps.append(QueryStep(
                step_number=step_num,
                description=f"Apply {trans_type} transformation",
                query=f"VTK {trans_type} transformation",
                focus="transformation"
            ))
            step_num += 1
        
        # Step 5: Visualization (mapper + actor + renderer)
        # Add 3 visualization steps for most queries (unless explicitly print-only)
        if 'print' not in q and not any(t in q for t in ['save only', 'export only']):
            # Mapper step
            steps.append(QueryStep(
                step_number=step_num,
                description="Create mapper for visualization",
                query="VTK create mapper polydata",
                focus="visualization_setup"
            ))
            step_num += 1
            
            # Actor step
            steps.append(QueryStep(
                step_number=step_num,
                description="Create actor and set properties",
                query="VTK create actor set properties",
                focus="visualization_setup"
            ))
            step_num += 1
            
            # Renderer step
            steps.append(QueryStep(
                step_number=step_num,
                description="Render and display scene",
                query="VTK render window display scene",
                focus="visualization"
            ))
            step_num += 1
        
        # Step 6: Output/save (if present)
        if any(t in q for t in ['save', 'write', 'export', 'output to file']):
            # Extract output format if present
            formats = ['png', 'jpg', 'jpeg', 'vtk', 'vtp', 'vtu', 'stl', 'obj', 'xml']
            output_format = next((fmt for fmt in formats if fmt in q), 'file')
            
            steps.append(QueryStep(
                step_number=step_num,
                description=f"Save/write to {output_format}",
                query=f"VTK write save {output_format} file",
                focus="data_io"
            ))
            step_num += 1
        
        # Ensure at least 1 step
        if not steps:
            steps.append(QueryStep(
                step_number=1,
                description="Execute VTK operation",
                query=query,
                focus="general"
            ))
        
        return steps
    
    def retrieve_for_step(
        self,
        step: QueryStep,
        top_k: int = 3
    ) -> StepResult:
        """
        Retrieve chunks for a single step
        
        Args:
            step: QueryStep to retrieve for
            top_k: Number of chunks to retrieve
            
        Returns:
            StepResult with retrieved chunks
        """
        # Retrieve code chunks for this step
        chunks = self.retriever.retrieve_code(
            step.query,
            top_k=top_k,
            prefer_pythonic=True,
            prefer_self_contained=True
        )
        
        # Calculate tokens
        token_count = self.retriever.estimate_total_tokens(chunks)
        
        return StepResult(
            step=step,
            retrieved_chunks=chunks,
            token_count=token_count
        )
    
    def process_query(self, query: str, **kwargs) -> Dict:
        """
        Main entry point: Process ANY query type
        
        Classifies query type and routes to appropriate handler:
        - code: Multi-step code generation (existing)
        - api: API documentation lookup
        - explanation: Concept explanation
        - data_query: Exploratory data→code (multiple techniques)
        - code_to_data: Find data files for code
        
        Args:
            query: User query
            **kwargs: Additional parameters for specific handlers
        
        Returns:
            Dict with JSON response
        """
        # Classify query type
        query_type = self._classify_query(query, **kwargs)
        
        # Route to appropriate handler
        handlers = {
            "code": self._handle_code_query,
            "api": self._handle_api_query,
            "explanation": self._handle_explanation_query,
            "data_query": self._handle_data_query,
            "code_to_data": self._handle_code_to_data_query,
            "refinement": self._handle_refinement_query,
        }
        
        handler = handlers.get(query_type, self._handle_code_query)
        return handler(query, **kwargs)
    
    def generate(
        self,
        query: str,
        top_k_per_step: int = 5
    ) -> PipelineResult:
        """
        Legacy method - calls _handle_code_query
        
        DEPRECATED: Use process_query() instead
        
        Args:
            query: User query
            top_k_per_step: Chunks to retrieve per step
        
        Returns:
            PipelineResult with code, explanation, and metadata
        """
        result = self._handle_code_query(query, top_k_per_step=top_k_per_step)
        # Convert dict to PipelineResult for backward compatibility
        return self._dict_to_pipeline_result(result)
    
    def _handle_code_query(
        self,
        query: str,
        top_k_per_step: int = 5
    ) -> Dict:
        """
        Handle CODE generation queries (multi-step decomposition)
        
        This is the original generate() method, now returns JSON
        
        Args:
            query: User query
            top_k_per_step: Chunks to retrieve per step
        
        Returns:
            Dict with JSON response
        """
        # Initialize prompt template for centralized prompting
        from prompt_templates import VTKPromptTemplate
        prompt_template = VTKPromptTemplate()
        
        # Step 1: Decompose into steps
        steps = self.decompose_query(query)
        
        # Step 2: Generate solutions step-by-step
        step_solutions = []
        all_chunks = []
        chunk_ids = []
        
        for step_idx, step in enumerate(steps, 1):
            # Retrieve for this step
            step_result = self.retrieve_for_step(step, top_k=top_k_per_step)
            step_chunks = step_result.retrieved_chunks
            
            # Track unique chunks
            for chunk in step_chunks:
                if chunk.chunk_id not in chunk_ids:
                    all_chunks.append(chunk)
                    chunk_ids.append(chunk.chunk_id)
            
            # Build structured generation input using JSON
            generation_input = {
                "original_query": query,
                "overall_understanding": f"Solving step {step_idx} of {len(steps)}",
                "overall_plan": {
                    "total_steps": len(steps),
                    "current_step_number": step.step_number,
                    "steps": [
                        {
                            "step_number": s.step_number,
                            "description": s.description,
                            "status": "completed" if i < step_idx - 1 else 
                                     "current" if i == step_idx - 1 else "pending"
                        }
                        for i, s in enumerate(steps)
                    ]
                },
                "current_step": {
                    "step_number": step.step_number,
                    "description": step.description,
                    "focus": step.focus
                },
                "previous_steps": [
                    {
                        "step_number": sol["step_number"],
                        "understanding": sol["understanding"],
                        "imports": sol["imports"],
                        "code": sol["code"]
                    }
                    for sol in step_solutions
                ],
                "documentation": [
                    {
                        "index": i + 1,
                        "chunk_id": c.chunk_id,
                        "content_type": c.content_type,
                        "content": c.content
                    }
                    for i, c in enumerate(step_chunks)
                ],
                "instructions": prompt_template.get_generation_instructions()
            }
            
            # Generate solution using JSON
            step_response = self.llm_client.generate_json(
                prompt_data=generation_input,
                schema_name="GenerationOutput",
                temperature=0.1
            )
            
            step_solutions.append({
                'step_number': step_response['step_number'],
                'understanding': step_response['understanding'],
                'imports': step_response['imports'],
                'code': step_response['code'],
                'citations': step_response.get('citations', []),
                'chunks_used': [c.chunk_id for c in step_chunks]
            })
        
        # Step 3: Assemble final result with proper import deduplication
        final_result = self._assemble_final_result(
            query=query,
            steps=steps,
            step_solutions=step_solutions,
            chunk_ids=chunk_ids
        )
        
        # Step 4: Security validation (ALWAYS performed)
        is_safe, security_issues = self.security_validator.validate_code(final_result.code)
        
        if not is_safe:
            logger.warning(f"Generated code failed security validation: {security_issues}")
        
        # Convert PipelineResult to JSON dict
        return {
            "response_type": "answer",
            "content_type": "code",
            "query": final_result.query,
            "code": final_result.code,
            "explanation": final_result.explanation,
            "citations": [{"number": i+1, "reason": f"Step {i+1} documentation"} for i in range(len(steps))],
            "chunk_ids_used": final_result.chunk_ids_used,
            "step_solutions": final_result.step_solutions,
            "validation_attempted": final_result.validation_attempted,
            "validation_errors_found": final_result.validation_errors_found,
            "api_validation_attempted": final_result.api_validation_attempted,
            "api_validation_passed": final_result.api_validation_passed,
            "api_validation_errors": final_result.api_validation_errors,
            "security_check_passed": is_safe,
            "security_issues": security_issues if not is_safe else [],
            "confidence": "high" if (final_result.validation_final_status == "passed" and is_safe and final_result.api_validation_passed) else "medium"
        }
    
    def _fix_common_vtk_issues(self, code: str) -> str:
        """
        Fix common VTK issues that prompts should prevent but may still occur
        
        This validation layer catches:
        1. Missing OpenGL2 import for rendering code
        2. VTK constants imported incorrectly (replace with numeric values)
        
        Strategy: Prompt first (preventive), fix if needed (corrective)
        
        Args:
            code: Python code with potential issues
            
        Returns:
            Code with fixes applied
        """
        # Fix 1: Add OpenGL2 import for rendering code if missing
        needs_opengl2 = (
            ('vtkRenderWindow' in code or 
             'vtkRenderer' in code or
             'vtkRenderWindowInteractor' in code or
             'vtkActor' in code) and
            'vtkRenderingOpenGL2' not in code
        )
        
        if needs_opengl2:
            # Find first import line to insert OpenGL2 before it
            lines = code.split('\n')
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    insert_pos = i
                    break
            
            # Insert OpenGL2 import at the beginning
            opengl2_import = 'import vtkmodules.vtkRenderingOpenGL2  # Auto-added for offscreen rendering'
            lines.insert(insert_pos, opengl2_import)
            code = '\n'.join(lines)
        
        # Fix 2: Replace VTK constant imports with numeric values
        # VTK scalar type constants should not be imported from specific modules
        vtk_constants = {
            'VTK_VOID': '0',
            'VTK_BIT': '1',
            'VTK_CHAR': '2',
            'VTK_UNSIGNED_CHAR': '3',
            'VTK_SHORT': '4',
            'VTK_UNSIGNED_SHORT': '5',
            'VTK_INT': '6',
            'VTK_UNSIGNED_INT': '7',
            'VTK_LONG': '8',
            'VTK_UNSIGNED_LONG': '9',
            'VTK_FLOAT': '10',
            'VTK_DOUBLE': '11',
            'VTK_ID_TYPE': '12',
        }
        
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            # Check if this line imports a VTK constant
            skip_line = False
            for const_name in vtk_constants.keys():
                if f'import {const_name}' in line and ('from vtkmodules.' in line or 'from vtk.' in line):
                    # Remove this import line
                    skip_line = True
                    logger.info(f"Removed VTK constant import: {line.strip()}")
                    break
            
            if not skip_line:
                # Replace constant usage with numeric value
                modified_line = line
                for const_name, numeric_value in vtk_constants.items():
                    if const_name in modified_line:
                        # Only replace if it's used as a standalone identifier (not part of another word)
                        import re
                        pattern = r'\b' + const_name + r'\b'
                        if re.search(pattern, modified_line):
                            modified_line = re.sub(pattern, numeric_value, modified_line)
                            logger.info(f"Replaced {const_name} with {numeric_value}")
                
                new_lines.append(modified_line)
        
        code = '\n'.join(new_lines)
        return code
    
    def _apply_api_fixes(self, code: str, validation_result) -> str:
        """
        Apply automatic API fixes when LLM validation fails
        
        This is a fallback for when the LLM cannot fix API errors.
        Strategy:
        1. High-confidence fixes (fuzzy match > 0.8): Auto-replace
        2. No suggestion: Delete problematic line
        
        Args:
            code: Python code with API errors
            validation_result: Result from api_validator.validate_code()
            
        Returns:
            Code with automatic fixes applied
        """
        if not validation_result or not validation_result.errors:
            return code
        
        lines = code.split('\n')
        fixes_applied = []
        
        for error in validation_result.errors:
            if error.error_type in ['method', 'import']:
                if error.suggestion:
                    # High-confidence fix: Replace with suggestion
                    if error.error_type == 'method' and error.line:
                        # Find and replace the method call line
                        for i, line in enumerate(lines):
                            if error.line in line:
                                # Extract method name from error (e.g., UpdateTimeStep -> SetTimeStep)
                                # Error line format: "reader.UpdateTimeStep(time_steps[10])"
                                old_method = error.line.split('(')[0].split('.')[-1] if '.' in error.line else None
                                if old_method:
                                    lines[i] = line.replace(old_method, error.suggestion)
                                    fixes_applied.append(f"Replaced {old_method}() with {error.suggestion}()")
                                break
                    
                    elif error.error_type == 'import' and error.line:
                        # Replace import line
                        for i, line in enumerate(lines):
                            if error.line.strip() == line.strip():
                                lines[i] = error.suggestion
                                fixes_applied.append(f"Fixed import: {error.suggestion}")
                                break
                else:
                    # No suggestion: Delete the problematic line
                    if error.line:
                        for i, line in enumerate(lines):
                            if error.line in line:
                                lines[i] = f"# REMOVED: {line}  # Invalid VTK API"
                                fixes_applied.append(f"Removed invalid line: {error.line}")
                                break
        
        if fixes_applied:
            logger.info(f"Applied {len(fixes_applied)} automatic API fixes:")
            for fix in fixes_applied:
                logger.info(f"  - {fix}")
        
        return '\n'.join(lines)
    
    def _validate_api_with_mcp(self, code: str) -> tuple[str, bool, List[str]]:
        """
        Validate code for VTK API hallucinations using MCP validator
        
        Catches:
        - Non-existent methods (e.g., SetOutputWholeExtent)
        - Non-existent classes
        - Wrong import modules
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (code, is_valid, errors_list)
        """
        if not self.api_validator:
            return code, True, []
        
        try:
            # Validate code
            result = self.api_validator.validate_code(code)
            
            if result.is_valid:
                return code, True, []
            
            # Code has API errors - format them for logging
            errors = []
            for error in result.errors:
                error_msg = f"{error.error_type}: {error.message}"
                if error.suggestion:
                    error_msg += f" (Suggestion: {error.suggestion})"
                errors.append(error_msg)
            
            logger.warning(f"API validation found {len(errors)} issue(s):")
            for error in errors:
                logger.warning(f"  - {error}")
            
            # Return code unchanged but flag validation failure
            # We don't auto-fix API errors (unlike VTK issues) - they need LLM review
            return code, False, errors
            
        except Exception as e:
            logger.error(f"API validation error: {e}")
            return code, True, []  # Fail gracefully
    
    def _assemble_final_result(
        self,
        query: str,
        steps: List[QueryStep],
        step_solutions: List[Dict],
        chunk_ids: List[str]
    ) -> PipelineResult:
        """
        Assemble final result from step solutions with import deduplication
        
        Args:
            query: Original query
            steps: List of query steps
            step_solutions: List of step solution dicts with imports, code, etc.
            chunk_ids: List of chunk IDs used
            
        Returns:
            PipelineResult with assembled code
        """
        if not step_solutions:
            return PipelineResult(
                query=query,
                steps=steps,
                code="",
                explanation="No solutions generated",
                chunk_ids_used=[],
                step_solutions=[],
                validation_attempted=False,
                validation_errors_found=0,
                validation_retries=0,
                validation_final_status="not_run",
                api_validation_attempted=False,
                api_validation_passed=True,
                api_validation_errors=[]
            )
        
        # Deduplicate imports across all steps
        all_imports = []
        seen_imports = set()
        for solution in step_solutions:
            for import_stmt in solution.get('imports', []):
                if import_stmt and import_stmt not in seen_imports:
                    all_imports.append(import_stmt)
                    seen_imports.add(import_stmt)
        
        # Concatenate code from all steps
        all_code = []
        for solution in step_solutions:
            code = solution.get('code', '').strip()
            if code:
                all_code.append(code)
        
        # Assemble final code: imports + code
        imports_section = "\n".join(all_imports)
        code_section = "\n\n".join(all_code)
        final_code = f"{imports_section}\n\n{code_section}".strip()
        
        # CRITICAL: Fix common VTK issues (prompts should prevent these, but fix if needed)
        final_code = self._fix_common_vtk_issues(final_code)
        
        # API Validation: Check for VTK API hallucinations (non-existent methods/classes)
        api_validation_attempted = False
        api_validation_passed = True
        api_validation_errors = []
        
        if self.api_validator and self.enable_api_validation:
            api_validation_attempted = True
            final_code, api_validation_passed, api_validation_errors = self._validate_api_with_mcp(final_code)
        
        # Validate and fix code if validation enabled
        validation_attempted = False
        validation_errors_found = 0
        validation_retries = 0
        validation_status = "not_run"
        changes_made = []
        
        if self.validator and self.enable_validation:
            validation_attempted = True
            context = f"Query: {query}\nGenerated code to solve: {steps[0].description if steps else 'N/A'}"
            
            # Include API validation errors in context so LLM can fix them
            if api_validation_errors:
                context += "\n\n⚠️ VTK API VALIDATION ERRORS DETECTED:\n"
                context += "The following VTK API issues were found and MUST be fixed:\n\n"
                for i, error in enumerate(api_validation_errors, 1):
                    context += f"{i}. {error}\n"
                context += "\nIMPORTANT: These are VTK API hallucinations (methods/classes that don't exist). "
                context += "You must fix these errors by using the correct VTK API. "
                context += "Check the suggestions provided and use valid VTK methods/classes only."
            
            fixed_code, changes, success = self.validator.validate_and_fix(
                code=final_code,
                context=context,
                max_retries=self.validation_max_retries
            )
            
            if changes:
                validation_errors_found = len(changes)
                validation_retries = len([c for c in changes if 'attempt' in str(c).lower()])
                final_code = fixed_code
                changes_made = changes
            
            validation_status = "passed" if success else "failed"
            
            # Re-validate with API validator if we had API errors and LLM fixed code
            if api_validation_errors and success and self.api_validator:
                logger.info("Re-validating with API validator after LLM fixes...")
                _, revalidation_passed, revalidation_errors = self._validate_api_with_mcp(final_code)
                
                if revalidation_passed:
                    logger.info("✅ API validation passed after LLM fixes")
                    api_validation_passed = True
                    api_validation_errors = []
                else:
                    logger.warning(f"⚠️ API validation still has {len(revalidation_errors)} error(s) after LLM fixes")
                    api_validation_errors = revalidation_errors
                    
                    # Step 2: LLM failed - apply automatic API fixes as fallback
                    logger.info("Applying automatic API fixes as fallback...")
                    revalidation_result = self.api_validator.validate_code(final_code)
                    final_code = self._apply_api_fixes(final_code, revalidation_result)
                    
                    # Re-validate after automatic fixes
                    _, auto_fix_passed, auto_fix_errors = self._validate_api_with_mcp(final_code)
                    if auto_fix_passed:
                        logger.info("✅ API validation passed after automatic fixes")
                        api_validation_passed = True
                        api_validation_errors = []
                    else:
                        logger.warning(f"⚠️ {len(auto_fix_errors)} error(s) remain after automatic fixes")
                        api_validation_errors = auto_fix_errors
        
        # Build explanation with code interspersed
        explanation_lines = [f"Query: {query}\n"]
        
        # Add imports section (no label)
        if all_imports:
            explanation_lines.append("```python")
            explanation_lines.append("\n".join(all_imports))
            explanation_lines.append("```\n")
        
        # Add each step with explanation and code (no labels)
        for i, solution in enumerate(step_solutions, 1):
            explanation_lines.append(solution.get('understanding', ''))
            
            # Add code for this step
            step_code = solution.get('code', '').strip()
            if step_code:
                explanation_lines.append("```python")
                explanation_lines.append(step_code)
                explanation_lines.append("```")
            explanation_lines.append("")  # Blank line between sections
        
        if changes_made:
            explanation_lines.append("\nValidation fixes applied:")
            for change in changes_made:
                explanation_lines.append(f"  - {change.get('error_type', 'unknown')}: {change.get('fix', 'N/A')}")
        
        explanation = "\n".join(explanation_lines)
        
        return PipelineResult(
            query=query,
            steps=steps,
            code=final_code,
            explanation=explanation,
            chunk_ids_used=chunk_ids,
            step_solutions=step_solutions,
            validation_attempted=validation_attempted,
            validation_errors_found=validation_errors_found,
            validation_retries=validation_retries,
            validation_final_status=validation_status,
            api_validation_attempted=api_validation_attempted,
            api_validation_passed=api_validation_passed,
            api_validation_errors=api_validation_errors
        )
    
    # ========== QUERY CLASSIFICATION AND ROUTING ==========
    
    def _classify_query(self, query: str, **kwargs) -> str:
        """
        Classify query type based on content and keywords
        
        Args:
            query: User query
            **kwargs: May contain 'code', 'data_file', 'existing_code', etc.
        
        Returns:
            Query type: "code", "api", "explanation", "data_query", "code_to_data", "refinement"
        """
        query_lower = query.lower()
        
        # Check for code refinement (existing_code provided)
        if 'existing_code' in kwargs and kwargs['existing_code']:
            return "refinement"
        
        # Check for explicit code provided (code_to_data)
        if 'code' in kwargs:
            return "code_to_data"
        
        # Data query indicators (exploratory)
        if any(ext in query_lower for ext in ['.csv', '.stl', '.vti', '.vtp', '.vtk', '.ply', '.obj']):
            # If asking "what can I do" or "what techniques"
            if any(phrase in query_lower for phrase in [
                'what can', 'what could', 'what should', 'what are the options',
                'options for', 'techniques for', 'how to use', 'ways to',
                'what techniques', 'techniques can i use', 'techniques to use'
            ]):
                return "data_query"
        
        # API query indicators (check first - more specific)
        if any(word in query_lower for word in [
            'method', 'function', 'class', 'parameter', 'setmapper', 'getproperty',
            '()', 'api', 'documentation'
        ]):
            # These are API-related terms
            if any(phrase in query_lower for phrase in [
                'what does', 'what is', 'how does', 'explain the'
            ]):
                return "api"
        
        # Explanation indicators (more general)
        if any(phrase in query_lower for phrase in [
            'explain', 'describe', 'difference between', 'relationship between',
            'compare', 'what is the', 'how does the'
        ]):
            if any(word in query_lower for word in [
                'pipeline', 'workflow', 'concept', 'architecture', 'works',
                'difference', 'relationship', 'between', 'versus', 'vs'
            ]):
                return "explanation"
        
        # Default to code generation (most queries)
        return "code"
    
    # ========== QUERY HANDLERS ==========
    
    def _handle_api_query(self, query: str, **kwargs) -> Dict:
        """
        Handle API documentation queries
        
        Returns JSON with API explanations
        """
        from prompt_templates import VTKPromptTemplate
        prompt_template = VTKPromptTemplate()
        
        # Retrieve API docs
        chunks = self.retriever.retrieve_api_doc(query, top_k=5)
        
        # Build prompt
        instructions = prompt_template.get_api_lookup_instructions()
        context = self._format_context(chunks)
        
        prompt_data = {
            "instructions": instructions,
            "query": query,
            "context": context
        }
        
        # Generate JSON response
        result = self.llm_client.generate_json(
            prompt_data=prompt_data,
            schema_name="APILookupOutput"
        )
        
        return result
    
    def _handle_explanation_query(self, query: str, **kwargs) -> Dict:
        """
        Handle concept explanation queries
        
        Returns JSON with detailed explanations
        """
        from prompt_templates import VTKPromptTemplate
        prompt_template = VTKPromptTemplate()
        
        # Retrieve explanations
        chunks = self.retriever.retrieve_explanation(query, top_k=5)
        
        # Build prompt
        instructions = prompt_template.get_explanation_instructions()
        context = self._format_context(chunks)
        
        prompt_data = {
            "instructions": instructions,
            "query": query,
            "context": context
        }
        
        # Generate JSON response
        result = self.llm_client.generate_json(
            prompt_data=prompt_data,
            schema_name="ExplanationOutput"
        )
        
        return result
    
    def _handle_data_query(self, query: str, data_file: str = None, **kwargs) -> Dict:
        """
        Handle exploratory data query - suggests multiple techniques
        
        User asks: "I have points.csv, what can I do?"
        Returns: Working code + alternative approaches
        """
        from prompt_templates import VTKPromptTemplate
        from collections import defaultdict
        import re
        
        prompt_template = VTKPromptTemplate()
        
        # Extract file type
        if data_file:
            file_type = data_file.split('.')[-1]
        else:
            match = re.search(r'\.(\w+)', query)
            file_type = match.group(1) if match else None
        
        # Retrieve examples with this file type
        search_query = f"{file_type} data visualization examples" if file_type else query
        chunks = self.retriever.retrieve_code(
            query=search_query,
            top_k=10,
            prefer_pythonic=True
        )
        
        # Filter for chunks with data files
        chunks_with_data = [
            c for c in chunks
            if c.metadata.get('has_data_files', False)
        ]
        
        if file_type:
            # Further filter by file type
            chunks_with_data = [
                c for c in chunks_with_data
                if any(file_type in df.get('filename', '').lower()
                       for df in c.metadata.get('data_files', []))
            ]
        
        # Group by category
        grouped = defaultdict(list)
        for chunk in chunks_with_data[:10]:
            category = chunk.metadata.get('category', 'Other')
            grouped[category].append(chunk)
        
        # Build prompt
        instructions = prompt_template.get_data_to_code_instructions()
        context = self._format_context(chunks_with_data)
        
        prompt_data = {
            "instructions": instructions,
            "query": query,
            "data_file": data_file or "user's data file",
            "file_type": file_type or "unknown",
            "context": context,
            "available_categories": list(grouped.keys())
        }
        
        # Generate JSON with multiple technique suggestions
        result = self.llm_client.generate_json(
            prompt_data=prompt_data,
            schema_name="DataToCodeOutput"
        )
        
        return result
    
    def _handle_code_to_data_query(self, query: str, code: str, **kwargs) -> Dict:
        """
        Handle code-to-data query - finds example data files
        
        User provides code, asks: "Do you have example data for this?"
        Returns: List of data files with download URLs
        """
        from prompt_templates import VTKPromptTemplate
        import re
        
        prompt_template = VTKPromptTemplate()
        
        # Parse code to identify requirements
        analysis = self._analyze_code_requirements(code)
        
        # Search for examples with matching data files
        chunks = self._retrieve_examples_with_data_files(
            reader_type=analysis['reader_type'],
            file_extensions=analysis['file_extensions'],
            top_k=10
        )
        
        # Extract data files
        data_files = self._extract_data_files_from_chunks(chunks)
        
        # Build prompt
        instructions = prompt_template.get_code_to_data_instructions()
        context = self._format_context(chunks)
        
        prompt_data = {
            "instructions": instructions,
            "query": query,
            "code": code,
            "code_analysis": analysis,
            "available_data_files": data_files,
            "context": context
        }
        
        # Generate JSON response
        result = self.llm_client.generate_json(
            prompt_data=prompt_data,
            schema_name="CodeToDataOutput"
        )
        
        return result
    
    # ========== HELPER METHODS ==========
    
    def _format_context(self, chunks: List) -> str:
        """Format chunks as numbered context for LLM"""
        return "\n\n".join([
            f"[{i}] {chunk.content}"
            for i, chunk in enumerate(chunks, 1)
        ])
    
    def _analyze_code_requirements(self, code: str) -> Dict:
        """
        Parse code to determine what data it needs
        
        Returns dict with reader type, file extensions, VTK classes
        """
        import re
        
        # Map VTK readers to file extensions
        reader_to_extension = {
            'vtkSTLReader': ['.stl'],
            'vtkPLYReader': ['.ply'],
            'vtkOBJReader': ['.obj'],
            'vtkXMLPolyDataReader': ['.vtp'],
            'vtkXMLImageDataReader': ['.vti'],
            'vtkXMLUnstructuredGridReader': ['.vtu'],
            'vtkDICOMImageReader': ['.dcm'],
            'vtkJPEGReader': ['.jpg', '.jpeg'],
            'vtkPNGReader': ['.png'],
        }
        
        # Find reader in code
        reader_type = None
        file_extensions = []
        for reader, extensions in reader_to_extension.items():
            if reader in code:
                reader_type = reader
                file_extensions = extensions
                break
        
        # Check for pandas CSV
        if 'pd.read_csv' in code or 'pandas.read_csv' in code:
            reader_type = 'pandas_csv'
            file_extensions = ['.csv']
        
        # Extract VTK classes
        vtk_classes = list(set(re.findall(r'\b(vtk[A-Z][a-zA-Z0-9]*)\b', code)))
        
        return {
            'reader_type': reader_type,
            'file_extensions': file_extensions,
            'vtk_classes': vtk_classes,
            'data_required': reader_type is not None
        }
    
    def _retrieve_examples_with_data_files(
        self,
        reader_type: str,
        file_extensions: List[str],
        top_k: int = 10
    ) -> List:
        """Find examples with matching reader type and data files"""
        if not reader_type:
            return []
        
        query = f"{reader_type} example with data file"
        
        results = self.retriever.client.search(
            collection_name=self.retriever.collection_name,
            query_vector=self.retriever.embedding_model.encode(query).tolist(),
            query_filter={
                "must": [
                    {"key": "content_type", "match": {"value": "code"}},
                    {"key": "metadata.has_data_files", "match": {"value": True}}
                ]
            },
            limit=top_k * 2
        )
        
        chunks = self.retriever._format_results(results)
        
        # Filter by file extension
        if file_extensions:
            chunks = [
                c for c in chunks
                if any(
                    any(ext in df.get('filename', '').lower() for ext in file_extensions)
                    for df in c.metadata.get('data_files', [])
                )
            ]
        
        return chunks[:top_k]
    
    def _extract_data_files_from_chunks(self, chunks: List) -> List[Dict]:
        """Extract unique data files with download URLs"""
        seen = set()
        data_files = []
        
        for chunk in chunks:
            files = chunk.metadata.get('data_files', [])
            download_info = chunk.metadata.get('data_download_info', [])
            
            for file_data, download_data in zip(files, download_info):
                filename = file_data.get('filename')
                if filename and filename not in seen:
                    seen.add(filename)
                    data_files.append({
                        'filename': filename,
                        'description': file_data.get('description', 'Data file'),
                        'source_example': chunk.metadata.get('title', 'Unknown'),
                        'download_url': download_data.get('url') if download_data else None,
                        'file_size': download_data.get('size', 'Unknown'),
                        'file_type': filename.split('.')[-1].upper()
                    })
        
        return data_files
    
    def _dict_to_pipeline_result(self, result_dict: Dict) -> PipelineResult:
        """Convert JSON dict to PipelineResult for backward compatibility"""
        return PipelineResult(
            query=result_dict.get('query', ''),
            steps=[],
            code=result_dict.get('code', ''),
            explanation=result_dict.get('explanation', ''),
            chunk_ids_used=[],
            step_solutions=[]
        )
    
    # ========== CODE REFINEMENT ==========
    
    def _handle_refinement_query(
        self,
        query: str,
        existing_code: str,
        top_k_per_step: int = 5,
        explanation_style: str = "progressive"
    ) -> Dict:
        """
        Handle code refinement requests - modify existing code instead of regenerating
        
        Full implementation with 3 phases:
        - Phase 1: Core refinement (analysis, modification, assembly)
        - Phase 2: Smart decomposition (multi-step, selective retrieval)
        - Phase 3: Advanced features (diff, validation)
        
        Args:
            query: Modification request (e.g., "Increase resolution and make it blue")
            existing_code: Current VTK code to modify
            top_k_per_step: Docs to retrieve per modification step
            explanation_style: "progressive" or "diff"
        
        Returns:
            Dict with refinement result (JSON response)
        """
        from prompt_templates import VTKPromptTemplate
        prompt_template = VTKPromptTemplate()
        
        # Phase 1: Analyze existing code
        code_analysis = self._analyze_existing_code(existing_code)
        
        # Phase 2: Decompose modification request into steps
        modification_steps = self._decompose_modifications(query, code_analysis)
        
        # Phase 2: Determine if retrieval is needed for any step
        needs_retrieval = any(
            step.get('requires_retrieval', False) 
            for step in modification_steps
        )
        
        # Phase 2: Selective retrieval (only if needed)
        retrieved_chunks = []
        chunk_ids = []
        if needs_retrieval:
            for step in modification_steps:
                if step.get('requires_retrieval', False):
                    # Retrieve docs for this modification
                    step_result = self.retriever.retrieve_code(
                        query=step['description'],
                        top_k=top_k_per_step
                    )
                    for chunk in step_result:
                        if chunk.chunk_id not in chunk_ids:
                            retrieved_chunks.append(chunk)
                            chunk_ids.append(chunk.chunk_id)
        
        # Phase 1 & 2: Generate code modification with LLM
        modification_result = self._generate_code_modification(
            modification_request=query,
            current_code=existing_code,
            code_analysis=code_analysis,
            modification_steps=modification_steps,
            documentation=retrieved_chunks
        )
        
        # Phase 3: Validate modified code
        updated_code = modification_result['updated_code']
        validation_passed = self._validate_modified_code(updated_code, existing_code)
        
        # Phase 3: Generate diff
        diff = self._generate_diff(existing_code, updated_code, modification_result['modifications'])
        
        # Phase 1: Build explanation based on style
        if explanation_style == "progressive":
            explanation = self._build_progressive_explanation(
                query=query,
                modifications=modification_result['modifications'],
                new_imports=modification_result.get('new_imports', [])
            )
        elif explanation_style == "diff":
            explanation = self._build_diff_explanation(
                query=query,
                modifications=modification_result['modifications'],
                diff=diff
            )
        else:
            # Fallback: simple explanation
            explanation = f"Modified code based on request: {query}"
        
        # Build citations
        citations = [
            {
                "number": i + 1,
                "chunk_id": chunk.chunk_id,
                "reason": f"Documentation for modification step {i+1}"
            }
            for i, chunk in enumerate(retrieved_chunks)
        ]
        
        # Security validation (ALWAYS performed)
        is_safe, security_issues = self.security_validator.validate_code(updated_code)
        
        if not is_safe:
            logger.warning(f"Refined code failed security validation: {security_issues}")
        
        # Return refinement result
        return {
            "response_type": "answer",
            "content_type": "code_refinement",
            "query": query,
            "original_code": existing_code,
            "code": updated_code,  # For consistency with generation
            "explanation": explanation,
            "modifications": modification_result['modifications'],
            "new_imports": modification_result.get('new_imports', []),
            "citations": citations,
            "chunk_ids_used": chunk_ids,
            "security_check_passed": is_safe,
            "security_issues": security_issues if not is_safe else [],
            "confidence": "high" if (validation_passed and is_safe) else "medium",
            "diff": diff,
            "validation_passed": validation_passed
        }
    
    def _analyze_existing_code(self, code: str) -> Dict:
        """
        Phase 1: Analyze existing code structure
        
        Extracts:
        - VTK classes used
        - Variable names and assignments
        - Import statements
        - Code structure (has main, etc.)
        
        Args:
            code: Existing VTK code
        
        Returns:
            Dict with code analysis
        """
        # Extract VTK classes
        vtk_classes = list(set(re.findall(r'vtk[A-Z]\w+', code)))
        
        # Extract variable assignments (variable = vtkClass())
        variables = {}
        for match in re.finditer(r'(\w+)\s*=\s*(vtk[A-Z]\w+)\(', code):
            var_name, class_name = match.groups()
            variables[var_name] = class_name
        
        # Extract imports
        import_lines = [line.strip() for line in code.split('\n') if 'import' in line]
        
        # Check structure
        has_main = 'def main' in code or 'if __name__' in code
        
        # Extract method calls (for understanding existing configuration)
        method_calls = re.findall(r'\.([A-Z]\w+)\(', code)
        
        return {
            'vtk_classes': vtk_classes,
            'variables': variables,
            'imports': import_lines,
            'has_main': has_main,
            'method_calls': list(set(method_calls)),
            'code_length': len(code),
            'has_colors': 'vtkNamedColors' in code,
            'has_rendering': any(cls in vtk_classes for cls in ['vtkRenderer', 'vtkRenderWindow'])
        }
    
    def _decompose_modifications(self, query: str, code_analysis: Dict) -> List[Dict]:
        """
        Phase 2: Decompose modification request into steps using LLM
        
        Examples:
        - "Increase resolution and make it blue" → 2 steps
        - "Add texture mapping" → 1 step (requires retrieval)
        - "Make it red" → 1 step (no retrieval needed)
        
        Args:
            query: Modification request
            code_analysis: Analysis of existing code
        
        Returns:
            List of modification steps with {step_number, description, requires_retrieval}
        """
        # Build prompt for LLM decomposition
        prompt_data = {
            "task": "decompose_code_modification",
            "modification_request": query,
            "existing_code_structure": code_analysis,
            "instructions": """
            Break the modification request into sequential steps.
            For each step, determine if it requires documentation retrieval.
            
            Simple property changes (color, resolution, size) don't need retrieval.
            New features (textures, filters, readers) need retrieval.
            
            Return JSON with:
            {
                "understanding": "What the user wants to change",
                "modification_steps": [
                    {
                        "step_number": 1,
                        "description": "Step description",
                        "requires_retrieval": true/false
                    }
                ],
                "preserved_elements": ["What should NOT change"]
            }
            """
        }
        
        try:
            # Use LLM to decompose
            result = self.llm_client.generate_json(
                prompt_data=prompt_data,
                schema_name="ModificationDecompositionOutput",
                temperature=0.1
            )
            
            return result.get('modification_steps', [])
            
        except Exception as e:
            # Fallback: single-step modification
            return [{
                "step_number": 1,
                "description": query,
                "requires_retrieval": True  # Safe default
            }]
    
    def _generate_code_modification(
        self,
        modification_request: str,
        current_code: str,
        code_analysis: Dict,
        modification_steps: List[Dict],
        documentation: List
    ) -> Dict:
        """
        Phase 1 & 2: Generate code modification using LLM
        
        Args:
            modification_request: Original modification request
            current_code: Current code
            code_analysis: Code analysis results
            modification_steps: Decomposed modification steps
            documentation: Retrieved documentation chunks
        
        Returns:
            Dict with modifications, updated_code, new_imports
        """
        from prompt_templates import VTKPromptTemplate
        prompt_template = VTKPromptTemplate()
        
        # Build modification prompt
        modification_prompt = {
            "task": "modify_vtk_code",
            "modification_request": modification_request,
            "current_code": current_code,
            "code_structure": code_analysis,
            "modification_steps": [step['description'] for step in modification_steps],
            "documentation": [
                {
                    "index": i + 1,
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content
                }
                for i, chunk in enumerate(documentation)
            ],
            "instructions": """
            Modify the existing code to fulfill the request.
            
            Rules:
            1. Keep existing structure and variable names
            2. Only change what's necessary
            3. Preserve pythonic API style if already used
            4. Add imports only if needed
            5. For each modification, explain what changed and why
            
            Return JSON with:
            {
                "modifications": [
                    {
                        "step_number": 1,
                        "modification": "What was modified",
                        "explanation": "Why this change was made",
                        "code_changed": "Line that was changed (if applicable)",
                        "code_added": "Code that was added (if applicable)",
                        "variable_affected": "Variable name modified"
                    }
                ],
                "updated_code": "Complete updated code",
                "new_imports": ["New import statements"],
                "preserved_structure": true,
                "diff_summary": "Human-readable summary of changes"
            }
            """
        }
        
        try:
            # Generate modification with LLM
            result = self.llm_client.generate_json(
                prompt_data=modification_prompt,
                schema_name="CodeModificationOutput",
                temperature=0.1
            )
            
            return result
            
        except Exception as e:
            # Fallback: return current code with error
            return {
                "modifications": [{
                    "step_number": 1,
                    "modification": "Error during modification",
                    "explanation": f"Failed to modify code: {str(e)}",
                    "code_changed": "",
                    "code_added": "",
                    "variable_affected": ""
                }],
                "updated_code": current_code,
                "new_imports": [],
                "preserved_structure": True,
                "diff_summary": "No changes made due to error"
            }
    
    def _validate_modified_code(self, updated_code: str, original_code: str) -> bool:
        """
        Phase 3: Validate modified code
        
        Checks:
        - Syntax is valid
        - Imports are present
        - Key structure preserved
        
        Args:
            updated_code: Modified code
            original_code: Original code
        
        Returns:
            True if validation passed
        """
        try:
            # Check syntax
            compile(updated_code, '<string>', 'exec')
            
            # Check has imports
            has_imports = 'import' in updated_code or 'from' in updated_code
            
            # Basic validation passed
            return has_imports
            
        except SyntaxError:
            return False
    
    def _generate_diff(
        self,
        original_code: str,
        updated_code: str,
        modifications: List[Dict]
    ) -> str:
        """
        Phase 3: Generate diff showing changes
        
        Args:
            original_code: Original code
            updated_code: Updated code
            modifications: List of modifications
        
        Returns:
            Diff string in unified diff format
        """
        import difflib
        
        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            original_code.splitlines(keepends=True),
            updated_code.splitlines(keepends=True),
            fromfile='original',
            tofile='modified',
            lineterm=''
        ))
        
        if diff_lines:
            return ''.join(diff_lines)
        else:
            return "No changes detected"
    
    def _build_progressive_explanation(
        self,
        query: str,
        modifications: List[Dict],
        new_imports: List[str]
    ) -> str:
        """
        Phase 1: Build progressive explanation (step-by-step narrative)
        
        Args:
            query: Modification request
            modifications: List of modifications
            new_imports: New imports added
        
        Returns:
            Progressive explanation string
        """
        lines = []
        lines.append(f"Modification Request: {query}\n")
        
        # New imports (if any)
        if new_imports:
            lines.append("New imports added:")
            lines.append("```python")
            for imp in new_imports:
                lines.append(imp)
            lines.append("```\n")
        
        # Each modification step
        for mod in modifications:
            # Step explanation
            lines.append(mod['explanation'])
            
            # Show specific code change
            code_to_show = mod.get('code_changed') or mod.get('code_added', '')
            if code_to_show:
                lines.append("```python")
                lines.append(code_to_show)
                lines.append("```")
            
            lines.append("")  # Blank line
        
        # Summary
        if len(modifications) > 1:
            lines.append(f"All {len(modifications)} modifications have been applied to the code.")
        
        return "\n".join(lines)
    
    def _build_diff_explanation(
        self,
        query: str,
        modifications: List[Dict],
        diff: str
    ) -> str:
        """
        Phase 1: Build diff-style explanation
        
        Args:
            query: Modification request
            modifications: List of modifications
            diff: Diff string
        
        Returns:
            Diff-style explanation
        """
        lines = []
        lines.append(f"Modification Request: {query}\n")
        lines.append("Changes made:\n")
        
        for mod in modifications:
            lines.append(f"**{mod['modification']}:**")
            lines.append(mod['explanation'])
            lines.append("")
        
        lines.append("Diff:")
        lines.append("```diff")
        lines.append(diff)
        lines.append("```")
        
        return "\n".join(lines)
    
    # ========== SESSION-BASED REFINEMENT (with Undo/Rollback) ==========
    
    def create_refinement_session(self, initial_code: str, session_id: Optional[str] = None) -> RefinementSession:
        """
        Create a new refinement session with undo/rollback support.
        
        GUI Usage:
            # Start a refinement session
            session = pipeline.create_refinement_session(original_code)
            
            # Store session in GUI state
            self.current_session = session
        
        Args:
            initial_code: Starting code before any refinements
            session_id: Optional session identifier
        
        Returns:
            RefinementSession: Session object with history tracking
        """
        return RefinementSession(initial_code, session_id)
    
    def refine_in_session(
        self,
        session: RefinementSession,
        query: str,
        **kwargs
    ) -> Dict:
        """
        Refine code within a session (automatically tracks history).
        
        GUI Usage:
            # User requests modification
            result = pipeline.refine_in_session(
                self.current_session,
                user_query
            )
            
            # Session automatically stores the refinement
            # Update GUI with result['code']
            code_editor.setText(result['code'])
            
            # Enable/disable undo/redo buttons
            undo_btn.setEnabled(self.current_session.can_undo())
            redo_btn.setEnabled(self.current_session.can_redo())
        
        Args:
            session: RefinementSession object
            query: Modification request
            **kwargs: Additional arguments for refinement
        
        Returns:
            Dict: Refinement result with added 'session_version' field
        """
        # Get current code from session
        current_code = session.get_current_code()
        
        # Perform refinement using existing refinement handler
        result = self.process_query(
            query=query,
            existing_code=current_code,
            **kwargs
        )
        
        # Add to session history
        version = session.add_refinement(query, result)
        
        # Add session info to result for GUI display
        result['session_version'] = version
        result['session_info'] = session.get_session_info()
        
        return result
    
    def get_session_code(self, session: RefinementSession) -> str:
        """
        Get current code from session.
        
        GUI Usage:
            current_code = pipeline.get_session_code(self.current_session)
        
        Args:
            session: RefinementSession object
        
        Returns:
            str: Current code in session
        """
        return session.get_current_code()
    
    def undo_refinement(self, session: RefinementSession) -> str:
        """
        Undo last refinement in session.
        
        GUI Usage:
            # Undo button clicked
            try:
                previous_code = pipeline.undo_refinement(self.current_session)
                code_editor.setText(previous_code)
                self.update_undo_redo_buttons()
            except ValueError as e:
                # Already at initial version
                show_message("Cannot undo: already at initial version")
        
        Args:
            session: RefinementSession object
        
        Returns:
            str: Code after undo
        
        Raises:
            ValueError: If already at initial version
        """
        return session.undo()
    
    def redo_refinement(self, session: RefinementSession) -> str:
        """
        Redo refinement in session (after undo).
        
        GUI Usage:
            # Redo button clicked
            try:
                next_code = pipeline.redo_refinement(self.current_session)
                code_editor.setText(next_code)
                self.update_undo_redo_buttons()
            except ValueError as e:
                # Already at latest version
                show_message("Cannot redo: already at latest version")
        
        Args:
            session: RefinementSession object
        
        Returns:
            str: Code after redo
        
        Raises:
            ValueError: If already at latest version
        """
        return session.redo()
    
    def get_session_versions(self, session: RefinementSession) -> List[Dict]:
        """
        Get list of all versions in session for display.
        
        GUI Usage:
            # Populate version history list
            versions = pipeline.get_session_versions(self.current_session)
            
            version_list.clear()
            for v in versions:
                is_current = " (current)" if v['is_current'] else ""
                item = f"v{v['version']}: {v['query']}{is_current}"
                version_list.addItem(item)
        
        Args:
            session: RefinementSession object
        
        Returns:
            List[dict]: List of version summaries
        """
        return session.get_version_list()
