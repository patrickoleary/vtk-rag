#!/usr/bin/env python3
"""
Response Enricher for VTK RAG Pipeline

Enriches CODE responses with:
- Generated code explanations (via LLM)
- Data file requirements
- Baseline images

For API/EXPLANATION/IMAGE responses, passes through without enrichment.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'llm-generation'))
sys.path.append(str(Path(__file__).parent.parent / 'grounding-prompting'))
sys.path.append(str(Path(__file__).parent.parent / 'retrieval-pipeline'))

from response_parser import ParsedResponse, ResponseType
from generator import VTKRAGGenerator
from prompt_templates import VTKPromptTemplate, PromptConfig


@dataclass
class EnrichedResponse:
    """
    Enriched response with additional context for CODE queries
    
    For CODE responses:
    - parsed: Original parsed response (code + citations)
    - code_explanation: Generated explanation of the code
    - data_files: Required data files (if any)
    - baseline_image: Expected output image (if available)
    
    For API/EXPLANATION/IMAGE responses:
    - Just passes through the parsed response
    """
    parsed: ParsedResponse
    content_type: str  # 'code', 'api', 'explanation', 'image'
    
    # Enrichments (only for CODE)
    code_explanation: Optional[str] = None
    has_enrichment: bool = False
    
    def needs_enrichment(self) -> bool:
        """Check if this response type needs enrichment"""
        return (
            self.content_type == 'code' and 
            self.parsed.response_type == ResponseType.ANSWER and
            self.parsed.has_code
        )
    
    def get_display_sections(self) -> Dict[str, Any]:
        """Get all sections for UI display"""
        sections = {
            'response_type': self.parsed.response_type.value,
            'content_type': self.content_type
        }
        
        # Always include raw response
        sections['raw_answer'] = self.parsed.raw_text
        
        if self.parsed.response_type == ResponseType.ANSWER:
            # Code section
            if self.parsed.has_code:
                sections['code'] = self.parsed.get_main_code()
                sections['vtk_classes'] = self.parsed.get_all_vtk_classes()
            
            # Explanation section (original or generated)
            if self.code_explanation:
                sections['explanation'] = self.code_explanation
            elif self.parsed.explanations:
                sections['explanation'] = '\n\n'.join(self.parsed.explanations)
            
            # Data files section
            if self.parsed.has_data_files():
                sections['data_files'] = self.parsed.format_data_section()
            
            # Baseline image section
            if self.parsed.has_baseline_images():
                sections['baseline_image'] = self.parsed.format_baseline_section()
            
            # Citations
            if self.parsed.has_citations:
                sections['citations'] = [
                    {'number': c.number, 'chunk_id': c.chunk_id, 'source_type': c.source_type}
                    for c in self.parsed.citations
                ]
        
        elif self.parsed.response_type == ResponseType.CLARIFYING_QUESTION:
            sections['question'] = self.parsed.question
            sections['options'] = self.parsed.options
        
        elif self.parsed.response_type == ResponseType.REFUSAL:
            sections['refusal_reason'] = self.parsed.refusal_reason
        
        return sections


class ResponseEnricher:
    """
    Enrich parsed responses based on content type
    
    Usage:
        enricher = ResponseEnricher()
        
        # For CODE responses - generates explanation + extracts data/images
        enriched = enricher.enrich(parsed, content_type='code', metadata=metadata)
        
        # For API/EXPLANATION/IMAGE - passes through
        enriched = enricher.enrich(parsed, content_type='api')
    """
    
    def __init__(
        self, 
        generator: Optional[VTKRAGGenerator] = None,
        retriever = None
    ):
        """
        Initialize enricher
        
        Args:
            generator: LLM generator for code explanations (creates if None)
            retriever: TaskSpecificRetriever for API docs (creates if None)
        """
        self.generator = generator
        self.retriever = retriever
        self.prompt_template = VTKPromptTemplate(
            config=PromptConfig(
                require_citations=False,  # Explanation doesn't need citations
                allow_speculation=False
            )
        )
    
    def enrich(
        self, 
        parsed: ParsedResponse, 
        content_type: str,
        metadata: Optional[Dict] = None,
        generate_explanation: bool = True
    ) -> EnrichedResponse:
        """
        Enrich parsed response based on content type
        
        Args:
            parsed: Parsed LLM response
            content_type: 'code', 'api', 'explanation', or 'image'
            metadata: Original retrieval metadata
            generate_explanation: Generate code explanation (CODE only)
        
        Returns:
            EnrichedResponse with additional context
        """
        enriched = EnrichedResponse(
            parsed=parsed,
            content_type=content_type
        )
        
        # Only enrich CODE responses with ANSWER type
        if not enriched.needs_enrichment():
            return enriched
        
        # Generate code explanation
        if generate_explanation and parsed.has_code:
            explanation = self._generate_code_explanation(parsed, metadata)
            if explanation:
                enriched.code_explanation = explanation
                enriched.has_enrichment = True
        
        return enriched
    
    def _generate_code_explanation(
        self, 
        parsed: ParsedResponse,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Generate explanation of the code using LLM with API documentation
        
        Retrieves API docs for VTK classes used in the code, then passes both
        code and API docs to LLM for a grounded explanation.
        
        Token budget: ~1,700 tokens total
        - Code: ~400t
        - API docs: ~600t (3 chunks × 200t)
        - Prompt: ~200t
        - Response: ~500t
        
        Args:
            parsed: Parsed response with code
            metadata: Original metadata
        
        Returns:
            Generated explanation or None
        """
        if not parsed.has_code:
            return None
        
        # Get the main code block
        code = parsed.get_main_code()
        if not code:
            return None
        
        # Extract VTK classes from the code
        vtk_classes = parsed.get_all_vtk_classes()
        
        # Retrieve API docs for the VTK classes (if available)
        api_context = ""
        if vtk_classes and self.retriever is None:
            # Lazy-load retriever if needed
            try:
                from task_specific_retriever import TaskSpecificRetriever
                self.retriever = TaskSpecificRetriever()
            except Exception:
                pass  # Continue without API docs if retriever unavailable
        
        if vtk_classes and self.retriever:
            try:
                # Retrieve API docs for the specific classes
                # Limit to 3 chunks (~600 tokens) for efficiency
                api_results = self.retriever.retrieve_api_doc(
                    query=f"API documentation for {', '.join(vtk_classes[:5])}",  # Limit to first 5 classes
                    top_k=3
                )
                
                if api_results:
                    api_parts = []
                    for i, result in enumerate(api_results, 1):
                        api_parts.append(f"[API {i}] {result.content[:400]}...")  # Truncate to 400 chars
                    api_context = "\n\n".join(api_parts)
            except Exception as e:
                # Continue without API docs if retrieval fails
                print(f"  (API doc retrieval skipped: {e})")
        
        # Create explanation prompt (includes code + API docs)
        if api_context:
            prompt = f"""Explain this VTK code using the provided API documentation:

```python
{code}
```

API Documentation:
{api_context}

Provide:
1. What the code does (1-2 sentences)
2. Key VTK classes with brief descriptions from API docs
3. Important methods called and their parameters
4. Any key settings or configurations

Keep it focused and practical for users learning VTK."""
        else:
            # Fallback: explain without API docs
            prompt = f"""Explain this VTK code in a clear, concise way:

```python
{code}
```

Provide:
1. What the code does (1-2 sentences)
2. Key VTK classes used and their purpose
3. Important parameters or settings

Keep it brief and focused on helping users understand the code."""
        
        try:
            # Initialize generator if needed
            if self.generator is None:
                from llm_client import LLMClient
                llm = LLMClient()
            else:
                # Access the generator's LLM client
                llm = self.generator.llm_client if hasattr(self.generator, 'llm_client') else LLMClient()
            
            # Generate explanation (direct LLM call, no RAG context)
            response = llm.generate(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"Warning: Could not generate code explanation: {e}")
            return None


def enrich_pipeline_response(
    parsed: ParsedResponse,
    content_type: str,
    metadata: Optional[Dict] = None,
    generate_explanation: bool = True
) -> EnrichedResponse:
    """
    Convenience function to enrich a pipeline response
    
    Args:
        parsed: Parsed LLM response
        content_type: 'code', 'api', 'explanation', or 'image'
        metadata: Original retrieval metadata
        generate_explanation: Generate code explanation for CODE responses
    
    Returns:
        EnrichedResponse ready for display
    """
    enricher = ResponseEnricher()
    return enricher.enrich(parsed, content_type, metadata, generate_explanation)


# Example usage
if __name__ == "__main__":
    from response_parser import ResponseParser
    
    print("=" * 80)
    print("Response Enricher - Example Usage")
    print("=" * 80)
    
    # Mock CODE response
    code_response = """To create a cylinder, use vtkCylinderSource [1]:

```python
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

cylinder = vtkCylinderSource()
cylinder.SetRadius(5.0)
cylinder.SetHeight(10.0)
cylinder.Update()

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)
```

This creates a cylinder with the specified dimensions [1]."""
    
    # Mock metadata
    metadata = {
        'query': 'Create a cylinder',
        'chunks_used': 2,
        'chunk_details': [
            {
                'position': 1,
                'chunk_id': 'CylinderExample_code_0',
                'source_type': 'example',
                'content_type': 'code',
                'score': 0.95,
                'metadata': {
                    'requires_data_files': False,
                    'image_url': 'https://examples.vtk.org/img/CylinderExample.png'
                }
            }
        ]
    }
    
    # Parse response
    parser = ResponseParser()
    parsed = parser.parse(code_response, metadata)
    
    print(f"\n✓ Parsed response:")
    print(f"  Type: {parsed.response_type.value}")
    print(f"  Has code: {parsed.has_code}")
    print(f"  Code blocks: {len(parsed.code_blocks)}")
    print(f"  VTK classes: {', '.join(parsed.get_all_vtk_classes())}")
    print(f"  Citations: {len(parsed.citations)}")
    
    # Enrich for CODE content type
    print(f"\n{'=' * 80}")
    print("Enriching CODE response...")
    print(f"{'=' * 80}")
    
    enricher = ResponseEnricher()
    enriched = enricher.enrich(
        parsed=parsed,
        content_type='code',
        metadata=metadata,
        generate_explanation=True  # Set to False if no API key
    )
    
    print(f"\n✓ Enriched response:")
    print(f"  Needs enrichment: {enriched.needs_enrichment()}")
    print(f"  Has enrichment: {enriched.has_enrichment}")
    
    # Display sections
    sections = enriched.get_display_sections()
    
    print(f"\n{'=' * 80}")
    print("Display Sections")
    print(f"{'=' * 80}")
    
    for section_name, section_content in sections.items():
        if section_content:
            print(f"\n## {section_name.upper()}")
            if isinstance(section_content, list):
                for item in section_content:
                    print(f"  - {item}")
            elif isinstance(section_content, str) and len(section_content) > 200:
                print(f"  {section_content[:200]}...")
            else:
                print(f"  {section_content}")
    
    print(f"\n{'=' * 80}")
    print("✓ Example complete!")
    print(f"{'=' * 80}")
