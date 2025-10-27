#!/usr/bin/env python3
"""
JSON Response Processor for VTK RAG

Processes JSON responses from the unified query system:
- Validates JSON structure
- Extracts VTK classes from code
- Enriches with metadata (optionally with LLM)
- Validates citations
- No text parsing required (everything is already JSON)

This replaces the old response_parser.py which did text parsing.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / 'llm-generation'))

try:
    from llm_client import LLMClient
    from prompt_templates import VTKPromptTemplate
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


@dataclass
class EnrichedResponse:
    """Enriched response with metadata"""
    original: Dict
    vtk_classes: List[str]
    has_code: bool
    has_citations: bool
    citation_count: int
    confidence: str
    response_type: str
    content_type: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)


class JSONResponseProcessor:
    """
    Process JSON responses from query handlers
    
    Takes structured JSON, validates, and enriches with metadata.
    No text parsing - works directly with JSON structure.
    """
    
    def __init__(self):
        """Initialize processor"""
        self.vtk_class_pattern = re.compile(r'\b(vtk[A-Z][a-zA-Z0-9]*)\b')
    
    def process(self, response: Dict) -> EnrichedResponse:
        """
        Process and enrich a JSON response
        
        Args:
            response: JSON dict from query handler
        
        Returns:
            EnrichedResponse with enriched metadata
        """
        # Validate basic structure
        self._validate_response(response)
        
        # Extract VTK classes
        vtk_classes = self._extract_vtk_classes(response)
        
        # Check for code
        has_code = 'code' in response and bool(response['code'])
        
        # Validate citations
        citations = response.get('citations', [])
        has_citations = len(citations) > 0
        citation_count = len(citations)
        
        # Get confidence
        confidence = response.get('confidence', 'unknown')
        
        # Build metadata
        metadata = {
            'processed': True,
            'vtk_class_count': len(vtk_classes),
            'unique_vtk_classes': len(set(vtk_classes)),
            'has_alternatives': 'alternative_approaches' in response,
            'has_data_files': 'data_files' in response,
        }
        
        # Add content-specific metadata
        if response.get('content_type') == 'code':
            metadata['code_length'] = len(response.get('code', ''))
            metadata['has_imports'] = 'import' in response.get('code', '')
        
        if response.get('content_type') == 'data':
            data_files = response.get('data_files', [])
            metadata['data_file_count'] = len(data_files)
            metadata['has_download_urls'] = all(
                df.get('download_url') for df in data_files
            )
        
        return EnrichedResponse(
            original=response,
            vtk_classes=vtk_classes,
            has_code=has_code,
            has_citations=has_citations,
            citation_count=citation_count,
            confidence=confidence,
            response_type=response.get('response_type', 'unknown'),
            content_type=response.get('content_type', 'unknown'),
            metadata=metadata
        )
    
    def _validate_response(self, response: Dict) -> None:
        """
        Validate JSON response structure
        
        Args:
            response: JSON dict to validate
        
        Raises:
            ValueError: If response is invalid
        """
        if not isinstance(response, dict):
            raise ValueError("Response must be a dictionary")
        
        # Check for response_type (should be present in all responses)
        if 'response_type' not in response:
            raise ValueError("Response missing 'response_type' field")
        
        # Validate response_type
        valid_types = ['answer', 'clarification', 'clarifying_question']
        if response['response_type'] not in valid_types:
            raise ValueError(
                f"Invalid response_type: {response['response_type']}. "
                f"Must be one of: {valid_types}"
            )
        
        # If answer, should have content_type
        if response['response_type'] == 'answer':
            if 'content_type' not in response:
                raise ValueError("Answer response missing 'content_type' field")
            
            valid_content = ['code', 'api', 'explanation', 'data', 'image']
            if response['content_type'] not in valid_content:
                raise ValueError(
                    f"Invalid content_type: {response['content_type']}. "
                    f"Must be one of: {valid_content}"
                )
    
    def _extract_vtk_classes(self, response: Dict) -> List[str]:
        """
        Extract VTK class names from response
        
        Looks in:
        - vtk_classes_used field (if present)
        - code field (if present)
        - explanation field
        
        Args:
            response: JSON response dict
        
        Returns:
            List of VTK class names found
        """
        classes: Set[str] = set()
        
        # Check explicit vtk_classes_used field
        if 'vtk_classes_used' in response:
            classes.update(response['vtk_classes_used'])
        
        # Extract from code
        if 'code' in response and response['code']:
            code_classes = self.vtk_class_pattern.findall(response['code'])
            classes.update(code_classes)
        
        # Extract from explanation
        if 'explanation' in response and response['explanation']:
            expl_classes = self.vtk_class_pattern.findall(response['explanation'])
            classes.update(expl_classes)
        
        # Extract from parameters (for API responses)
        if 'parameters' in response:
            for param in response['parameters']:
                if 'type' in param:
                    type_classes = self.vtk_class_pattern.findall(param['type'])
                    classes.update(type_classes)
        
        return sorted(list(classes))
    
    def validate_citations(self, response: Dict) -> Dict[str, any]:
        """
        Validate citation structure and numbering
        
        Args:
            response: JSON response dict
        
        Returns:
            Dict with validation results
        """
        citations = response.get('citations', [])
        
        if not citations:
            return {
                'valid': True,
                'warning': 'No citations present',
                'citation_count': 0
            }
        
        # Check citation structure
        issues = []
        seen_numbers = set()
        
        for i, citation in enumerate(citations):
            # Check required fields
            if 'number' not in citation:
                issues.append(f"Citation {i} missing 'number' field")
            else:
                num = citation['number']
                if num in seen_numbers:
                    issues.append(f"Duplicate citation number: {num}")
                seen_numbers.add(num)
            
            if 'reason' not in citation:
                issues.append(f"Citation {i} missing 'reason' field")
        
        # Check for sequential numbering
        if seen_numbers:
            expected = set(range(1, len(citations) + 1))
            if seen_numbers != expected:
                issues.append(
                    f"Citations not sequential: expected {expected}, got {seen_numbers}"
                )
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'citation_count': len(citations),
            'citation_numbers': sorted(list(seen_numbers))
        }
    
    def extract_mentioned_files(self, response: Dict) -> Dict[str, List[str]]:
        """
        Extract file mentions from response
        
        Args:
            response: JSON response dict
        
        Returns:
            Dict with categorized file mentions
        """
        files = {
            'data_files': [],
            'code_files': [],
            'mentioned_files': []
        }
        
        # Explicit data files
        if 'data_files' in response:
            files['data_files'] = [
                df.get('filename', '') for df in response['data_files']
            ]
        
        # Data files used
        if 'data_files_used' in response:
            files['data_files'].extend(response['data_files_used'])
        
        # Extract from code
        if 'code' in response and response['code']:
            # Look for common file extensions
            file_pattern = r'["\']([^"\']+\.(?:csv|stl|vti|vtp|vtk|ply|obj|jpg|png))["\']'
            code_files = re.findall(file_pattern, response['code'], re.IGNORECASE)
            files['code_files'] = code_files
        
        # Deduplicate
        files['data_files'] = list(set(files['data_files']))
        files['code_files'] = list(set(files['code_files']))
        
        return files
    
    def summarize(self, response: Dict) -> str:
        """
        Generate human-readable summary of response
        
        Args:
            response: JSON response dict
        
        Returns:
            Summary string
        """
        lines = []
        
        # Type
        resp_type = response.get('response_type', 'unknown')
        content_type = response.get('content_type', 'none')
        lines.append(f"Response: {resp_type} ({content_type})")
        
        # Confidence
        confidence = response.get('confidence', 'unknown')
        lines.append(f"Confidence: {confidence}")
        
        # VTK classes
        vtk_classes = self._extract_vtk_classes(response)
        if vtk_classes:
            lines.append(f"VTK Classes: {', '.join(vtk_classes[:5])}")
            if len(vtk_classes) > 5:
                lines.append(f"  (and {len(vtk_classes) - 5} more)")
        
        # Citations
        citations = response.get('citations', [])
        if citations:
            lines.append(f"Citations: {len(citations)} sources")
        
        # Code
        if 'code' in response and response['code']:
            code_lines = response['code'].count('\n') + 1
            lines.append(f"Code: {code_lines} lines")
        
        # Data files
        if 'data_files' in response:
            files = response['data_files']
            lines.append(f"Data files: {len(files)} available")
        
        # Alternatives
        if 'alternative_approaches' in response:
            alts = response['alternative_approaches']
            lines.append(f"Alternatives: {len(alts)} approaches")
        
        return '\n'.join(lines)
    
    def enrich_with_llm(
        self,
        response: Dict,
        documentation_chunks: List[Dict],
        llm_client: Optional['LLMClient'] = None
    ) -> Dict:
        """
        Enrich response with LLM-generated explanations
        
        For CODE responses:
        - If no explanation exists, generate one
        - If explanation exists but is poor, improve it
        
        For other response types:
        - Pass through unchanged (may enrich in future)
        
        Args:
            response: Original JSON response
            documentation_chunks: Documentation chunks for context
            llm_client: Optional LLM client (creates new one if not provided)
        
        Returns:
            Response with enriched explanation
        """
        if not HAS_LLM:
            return response  # Can't enrich without LLM support
        
        # Only enrich CODE responses
        content_type = response.get('content_type', '')
        if content_type != 'code':
            return response  # Pass through non-code responses
        
        # Check if we have code
        code = response.get('code', '')
        if not code:
            return response  # Nothing to explain
        
        # Initialize LLM client and templates
        if llm_client is None:
            llm_client = LLMClient()
        
        template = VTKPromptTemplate()
        
        # Determine if we need to generate or improve
        existing_explanation = response.get('explanation', '').strip()
        
        if not existing_explanation:
            # No explanation - generate one
            instructions = template.get_code_explanation_generation_instructions()
        else:
            # Has explanation - improve it
            instructions = template.get_explanation_improvement_instructions()
        
        # Format documentation context
        context = self._format_context_for_enrichment(documentation_chunks)
        
        # Build prompt
        prompt_data = {
            "instructions": instructions,
            "code": code,
            "existing_explanation": existing_explanation if existing_explanation else None,
            "documentation_context": context
        }
        
        try:
            # Get enriched explanation from LLM
            enrichment = llm_client.generate_json(
                prompt_data=prompt_data,
                schema_name="ExplanationEnrichmentOutput"
            )
            
            # Update response with improved explanation
            enriched_response = response.copy()
            enriched_response['explanation'] = enrichment['improved_explanation']
            
            # Add enrichment metadata
            enriched_response['_enrichment'] = {
                'was_enriched': True,
                'original_explanation': existing_explanation,
                'key_points': enrichment.get('key_points', []),
                'vtk_classes_explained': enrichment.get('vtk_classes_explained', []),
                'enrichment_confidence': enrichment.get('confidence', 'unknown')
            }
            
            return enriched_response
            
        except Exception as e:
            # If enrichment fails, return original
            print(f"Warning: Failed to enrich explanation: {e}")
            return response
    
    def _format_context_for_enrichment(self, chunks: List[Dict]) -> str:
        """
        Format documentation chunks for enrichment prompt
        
        Args:
            chunks: Documentation chunks
        
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No additional documentation available."
        
        lines = []
        for i, chunk in enumerate(chunks[:5], 1):  # Top 5 chunks
            content = chunk.get('content', chunk.get('text', ''))
            lines.append(f"[{i}] {content}")
        
        return '\n\n'.join(lines)


# Convenience function
def process_response(response: Dict) -> EnrichedResponse:
    """
    Process a JSON response (convenience function)
    
    Args:
        response: JSON dict from query handler
    
    Returns:
        EnrichedResponse with metadata
    """
    processor = JSONResponseProcessor()
    return processor.process(response)
