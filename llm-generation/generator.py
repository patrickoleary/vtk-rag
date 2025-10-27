#!/usr/bin/env python3
"""
VTK RAG Generator with Citation Enforcement

Generates LLM responses with:
- Forced citations
- No-answer if not in context
- Discouraged speculation
- Response validation

Integrates with grounding-prompting pipeline.
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add grounding-prompting to path
sys.path.append(str(Path(__file__).parent.parent / 'grounding-prompting'))

from llm_client import LLMClient, LLMConfig
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """Container for generated response with metadata"""
    answer: str
    citations_found: List[int]
    citation_count: int
    refused: bool
    metadata: Dict
    raw_response: str
    sources_used: str
    
    # Validation metrics
    validation_attempted: bool = False
    validation_errors_found: int = 0
    validation_retries: int = 0
    validation_final_status: str = "not_run"  # not_run, passed, failed


class VTKRAGGenerator:
    """
    RAG Generator with strict grounding enforcement
    
    Features:
    - Forces citations to retrieved chunks
    - Refuses when context insufficient
    - Validates responses
    - Tracks provenance
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        require_citations: bool = None,
        strict_grounding: bool = None,
        validate_code: bool = None,
        validation_max_retries: int = None
    ):
        """
        Initialize generator
        
        Args:
            llm_client: LLM client instance. If None, creates from env.
            require_citations: Require citations. If None, reads from env.
            strict_grounding: Enforce strict grounding. If None, reads from env.
            validate_code: Enable VTK code validation. If None, reads from env.
            validation_max_retries: Max correction attempts (0=disabled, 1-5 recommended). If None, reads from env.
        """
        self.client = llm_client or LLMClient()
        
        # Load settings from env
        self.require_citations = (
            require_citations if require_citations is not None 
            else os.getenv('REQUIRE_CITATIONS', 'true').lower() == 'true'
        )
        self.strict_grounding = (
            strict_grounding if strict_grounding is not None
            else os.getenv('STRICT_GROUNDING', 'true').lower() == 'true'
        )
        self.validate_code = (
            validate_code if validate_code is not None
            else os.getenv('VALIDATE_CODE', 'false').lower() == 'true'
        )
        self.validation_max_retries = (
            validation_max_retries if validation_max_retries is not None
            else int(os.getenv('VALIDATION_MAX_RETRIES', '1'))
        )
        
        # Load VTK validator if enabled
        self.validator = None
        if self.validate_code:
            try:
                sys.path.append(str(Path(__file__).parent.parent / 'api-mcp'))
                from vtk_validator import load_validator
                self.validator = load_validator()
                # VTK validation enabled
            except Exception as e:
                logger.warning(f"  Failed to load VTK validator: {e}")
                logger.warning(f"  Code validation disabled")
                self.validate_code = False
        
        # Generator initialized
    
    def generate(
        self,
        prompt: str,
        grounding_metadata: Dict,
        **kwargs
    ) -> GeneratedResponse:
        """
        Generate response with citation enforcement and optional code validation
        
        Args:
            prompt: Grounded prompt from grounding pipeline
            grounding_metadata: Metadata from grounding pipeline
            **kwargs: Additional generation parameters
        
        Returns:
            GeneratedResponse with answer and citations
        """
        # Validate metadata from grounding pipeline
        self._validate_metadata(grounding_metadata)
        
        # Generate response (initial attempt)
        raw_response = self.client.generate(prompt, **kwargs)
        
        # Code validation loop (if enabled)
        validation_stats = {
            'attempted': False,
            'errors_found': 0,
            'retries': 0,
            'final_status': 'not_run'
        }
        
        if self.validator and self.validation_max_retries > 0:
            raw_response, validation_stats = self._validate_and_correct_code(
                raw_response, 
                prompt,
                **kwargs
            )
        
        # Parse and validate citations
        citations = self._extract_citations(raw_response)
        refused = self._is_refusal(raw_response)
        
        # Validate citations if required
        if self.require_citations and not refused and not citations:
            logger.warning("Response lacks citations - this may be hallucination!")
        
        # Build response object
        response = GeneratedResponse(
            answer=raw_response,
            citations_found=citations,
            citation_count=len(citations),
            refused=refused,
            metadata=grounding_metadata,
            raw_response=raw_response,
            sources_used=self._format_sources_from_metadata(grounding_metadata),
            validation_attempted=validation_stats['attempted'],
            validation_errors_found=validation_stats['errors_found'],
            validation_retries=validation_stats['retries'],
            validation_final_status=validation_stats['final_status']
        )
        
        return response
    
    def _extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers [N] from text"""
        citations = re.findall(r'\[(\d+)\]', text)
        return sorted(list(set(int(c) for c in citations)))
    
    def _is_refusal(self, text: str) -> bool:
        """Check if response is a refusal"""
        refusal_phrases = [
            "don't have enough information",
            "not enough information",
            "cannot answer",
            "unable to answer",
            "insufficient information",
            "not in the context",
            "not in the provided",
            "consult the official documentation"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in refusal_phrases)
    
    def _validate_metadata(self, metadata: Dict) -> None:
        """
        Validate metadata from grounding pipeline
        
        Args:
            metadata: Grounding metadata dict
            
        Raises:
            ValueError: If required fields are missing
        """
        # Check for required fields
        required_fields = ['query', 'chunks_used', 'chunk_details']
        missing = [f for f in required_fields if f not in metadata]
        
        if missing:
            logger.warning(f"Metadata missing fields: {missing}")
            # Don't fail, just warn - metadata might be from older pipeline version
        
        # Validate chunk_details structure if present
        if 'chunk_details' in metadata:
            chunk_details = metadata['chunk_details']
            if not isinstance(chunk_details, list):
                logger.warning("chunk_details should be a list")
            elif len(chunk_details) > 0:
                # Check first chunk has expected structure
                first_chunk = chunk_details[0]
                expected_keys = ['position', 'chunk_id', 'source_type']
                missing_keys = [k for k in expected_keys if k not in first_chunk]
                if missing_keys:
                    logger.warning(f"chunk_details missing keys: {missing_keys}")
                
                # Log metadata enrichment info
                enriched_chunks = sum(1 for c in chunk_details 
                                     if 'data_files' in c or 'has_baseline' in c or 'user_query' in c or 'user_queries' in c)
                if enriched_chunks > 0:
                    pass  # Has enriched metadata
        
        # Log source distribution if available
        if 'source_distribution' in metadata:
            dist = metadata['source_distribution']
            pass  # Has source distribution
    
    def _format_sources_from_metadata(self, metadata: Dict) -> str:
        """Format source list from metadata"""
        if 'chunk_details' not in metadata:
            return "No source details available"
        
        lines = []
        for detail in metadata['chunk_details']:
            pos = detail['position']
            chunk_id = detail['chunk_id']
            source_type = detail['source_type']
            score = detail.get('score', 0.0)
            
            line = f"[{pos}] {chunk_id} ({source_type}) - Score: {score:.2f}"
            
            # Add title if available
            if 'title' in detail:
                line = f"[{pos}] {detail['title']} ({source_type}) - Score: {score:.2f}"
            elif 'class_name' in detail:
                line = f"[{pos}] {detail['class_name']} ({source_type}) - Score: {score:.2f}"
            
            lines.append(line)
        
        return '\n'.join(lines) if lines else "No sources"
    
    def validate_citations(
        self,
        response: GeneratedResponse,
        max_position: int
    ) -> Dict[str, any]:
        """
        Validate that citations are within bounds
        
        Args:
            response: Generated response
            max_position: Maximum valid citation number
        
        Returns:
            Validation results
        """
        valid_citations = []
        invalid_citations = []
        
        for cite in response.citations_found:
            if 1 <= cite <= max_position:
                valid_citations.append(cite)
            else:
                invalid_citations.append(cite)
        
        return {
            'all_valid': len(invalid_citations) == 0,
            'valid_citations': valid_citations,
            'invalid_citations': invalid_citations,
            'validation_passed': len(invalid_citations) == 0,
            'total_citations': len(response.citations_found)
        }
    
    def format_response(
        self,
        response: GeneratedResponse,
        include_sources: bool = True,
        include_metadata: bool = False
    ) -> str:
        """
        Format response for display
        
        Args:
            response: Generated response
            include_sources: Include source list
            include_metadata: Include metadata details
        
        Returns:
            Formatted response string
        """
        lines = []
        
        # Add answer
        lines.append("ANSWER:")
        lines.append(response.answer)
        
        # Add refusal indicator
        if response.refused:
            lines.append("\n⚠️  Response indicates insufficient information in context")
        
        # Add citations info
        if response.citation_count > 0:
            lines.append(f"\n✓ Citations: {response.citations_found}")
        elif not response.refused:
            lines.append("\n⚠️  No citations found - potential hallucination!")
        
        # Add sources
        if include_sources and response.sources_used:
            lines.append("\nSOURCES USED:")
            lines.append(response.sources_used)
        
        # Add metadata
        if include_metadata:
            lines.append("\nMETADATA:")
            lines.append(f"  Query: {response.metadata.get('query', 'N/A')}")
            lines.append(f"  Task: {response.metadata.get('task_type', 'N/A')}")
            lines.append(f"  Chunks used: {response.metadata.get('chunks_used', 0)}")
            lines.append(f"  Context tokens: {response.metadata.get('total_context_tokens', 0)}")
        
        return '\n'.join(lines)
    
    def _validate_and_correct_code(
        self,
        response: str,
        original_prompt: str,
        **kwargs
    ) -> Tuple[str, Dict]:
        """
        Validate code and attempt corrections if errors found
        
        Args:
            response: Initial LLM response
            original_prompt: Original prompt
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (corrected_response, validation_stats)
        """
        stats = {
            'attempted': True,
            'errors_found': 0,
            'retries': 0,
            'final_status': 'not_run'
        }
        
        # Extract code from response
        code = self._extract_code_from_response(response)
        if not code:
            logger.debug("No code found in response, skipping validation")
            stats['attempted'] = False
            return response, stats
        
        # Validate initial response
        validation = self.validator.validate_code(code)
        stats['errors_found'] = len(validation.errors)
        
        if not validation.has_errors:
            pass  # Validation passed
            stats['final_status'] = 'passed'
            return response, stats
        
        # Attempt corrections
        current_response = response
        for attempt in range(1, self.validation_max_retries + 1):
            pass  # Validation failed, retrying
            logger.debug(f"Errors:\n{validation.format_errors()}")
            
            stats['retries'] = attempt
            
            # Build correction prompt
            correction_prompt = self._build_correction_prompt(
                current_response,
                validation.errors,
                original_prompt
            )
            
            # Generate correction
            current_response = self.client.generate(correction_prompt, **kwargs)
            
            # Re-validate
            code = self._extract_code_from_response(current_response)
            if not code:
                logger.warning("No code found in corrected response")
                stats['final_status'] = 'failed'
                return current_response, stats
            
            validation = self.validator.validate_code(code)
            
            if not validation.has_errors:
                pass  # Validation passed after correction
                stats['final_status'] = 'passed'
                return current_response, stats
        
        # Max retries exhausted
        logger.warning(f"⚠️  Code validation failed after {self.validation_max_retries} attempts, returning last attempt")
        stats['final_status'] = 'failed'
        return current_response, stats
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response (code blocks)"""
        # Look for ```python ... ``` blocks
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            # Return first code block (usually the main one)
            return matches[0]
        
        # Try generic ``` blocks
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            # Check if it looks like Python (has import or def)
            for match in matches:
                if 'import' in match or 'def ' in match or 'class ' in match:
                    return match
        
        return None
    
    def _build_correction_prompt(
        self,
        response: str,
        errors: List,
        original_prompt: str
    ) -> str:
        """Build prompt for correcting validation errors"""
        from vtk_validator import ValidationError
        
        error_details = "\n".join([
            f"  {i+1}. {error.error_type.upper()}: {error.message}" +
            (f"\n     REQUIRED ACTION: {error.suggestion}" if error.suggestion else "")
            for i, error in enumerate(errors)
        ])
        
        # Extract just the code to keep prompt size manageable
        code = self._extract_code_from_response(response)
        if not code:
            code = response  # Fallback if no code block found
        
        return f"""The following VTK code has validation errors. You MUST fix ALL the errors listed below.

ERRORS THAT MUST BE FIXED:
{error_details}

CODE WITH ERRORS:
```python
{code}
```

INSTRUCTIONS:
1. Fix EVERY error listed above - these are not optional
2. Follow the REQUIRED ACTION for each error exactly
3. If the action says "DELETE": Remove the line completely - this is the SMALLEST change (less than trying to fix unused code)
4. If the action says "Replace": Change only that specific import/class/method
5. Keep all other code unchanged

Provide the corrected code in a ```python code block:"""
