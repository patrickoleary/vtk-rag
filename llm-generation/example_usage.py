#!/usr/bin/env python3
"""
Example Usage for LLM Generation

Demonstrates complete RAG pipeline:
1. Query rewriting
2. Vector retrieval
3. Reranking
4. Context grounding
5. LLM generation with citation enforcement

Shows different scenarios and configurations.
"""

import os
import sys
from pathlib import Path

# Add required modules to path
sys.path.append(str(Path(__file__).parent.parent / 'retrieval-pipeline'))
sys.path.append(str(Path(__file__).parent.parent / 'grounding-prompting'))

from generator import VTKRAGGenerator, GeneratedResponse
from llm_client import LLMClient, LLMConfig, LLMProvider


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def example1_mock_generation():
    """Example 1: Mock generation without real LLM"""
    print_section("Example 1: Mock Generation (No API Key Needed)")
    
    print("\nThis example shows the structure without calling real LLM")
    print("(Useful for testing without API keys)")
    
    # Mock response
    mock_response = """To create a cylinder with radius 5, use vtkCylinderSource [1]:

```python
from vtkmodules.vtkFiltersSources import vtkCylinderSource

cylinder = vtkCylinderSource()
cylinder.SetRadius(5.0)  # [1]
cylinder.SetHeight(10.0)
cylinder.Update()
```

As shown in the example [2], you can then connect it to a mapper for visualization."""
    
    # Mock metadata (from grounding pipeline)
    metadata = {
        'query': 'How to create cylinder with radius 5?',
        'task_type': 'code_generation',
        'chunks_used': 2,
        'total_context_tokens': 200,
        'chunk_details': [
            {
                'position': 1,
                'chunk_id': 'vtkCylinderSource_0',
                'source_type': 'api_doc',
                'class_name': 'vtkCylinderSource',
                'score': 8.5
            },
            {
                'position': 2,
                'chunk_id': 'CylinderExample_0',
                'source_type': 'example',
                'title': 'CylinderExample',
                'has_image': True,
                'score': 7.2
            }
        ]
    }
    
    # Create generator (won't actually call LLM in this example)
    from dataclasses import replace
    from generator import GeneratedResponse
    
    # Create mock response object
    generator = VTKRAGGenerator() if os.path.exists('.env') else None
    
    if generator:
        response = GeneratedResponse(
            answer=mock_response,
            citations_found=[1, 2],
            citation_count=2,
            refused=False,
            metadata=metadata,
            raw_response=mock_response,
            sources_used=generator._format_sources_from_metadata(metadata)
        )
        
        print("\n" + generator.format_response(response, include_sources=True))
    else:
        print("\n(Skipping - no .env file)")
        print(f"\nMock Response:\n{mock_response}")


def example2_refusal_response():
    """Example 2: Proper refusal when context insufficient"""
    print_section("Example 2: Refusal Policy Demonstration")
    
    print("\nWhen context doesn't contain the answer, LLM should refuse:")
    
    mock_response = """I don't have enough information in the provided VTK documentation to answer this question accurately. The context provided discusses cylinders, but your question is about creating cubes. Please consult the official VTK documentation or ask a more specific question about cylinders."""
    
    metadata = {
        'query': 'How to create a cube?',  # Not in context!
        'task_type': 'general',
        'chunks_used': 2,
        'chunk_details': [
            {'position': 1, 'chunk_id': 'vtkCylinderSource_0', 'source_type': 'api_doc', 'score': 3.2},
            {'position': 2, 'chunk_id': 'vtkSphereSource_0', 'source_type': 'api_doc', 'score': 2.8}
        ]
    }
    
    if os.path.exists('.env'):
        generator = VTKRAGGenerator()
        response = GeneratedResponse(
            answer=mock_response,
            citations_found=[],
            citation_count=0,
            refused=True,
            metadata=metadata,
            raw_response=mock_response,
            sources_used=generator._format_sources_from_metadata(metadata)
        )
        
        print("\n" + generator.format_response(response))
        print("\n✅ GOOD: LLM refused to answer instead of guessing!")
    else:
        print(f"\nMock Refusal:\n{mock_response}")
        print("\n✅ Refusal detected - no hallucination!")


def example3_configuration_options():
    """Example 3: Different LLM configurations"""
    print_section("Example 3: LLM Configuration Options")
    
    print("\nYou can configure different LLM providers in .env:")
    print("""
# OpenAI (GPT-4)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.1

# Anthropic (Claude)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-sonnet-20240229
ANTHROPIC_TEMPERATURE=0.1

# Local Model (via OpenAI-compatible API)
LLM_PROVIDER=local
LOCAL_API_BASE=http://localhost:8000/v1
LOCAL_MODEL=llama-2-70b
""")
    
    print("\nRecommended settings for VTK RAG:")
    print("  • Temperature: 0.1 (low = more focused, accurate)")
    print("  • Max tokens: 2000 (enough for code + explanation)")
    print("  • Require citations: true")
    print("  • Strict grounding: true")


def example4_citation_validation():
    """Example 4: Citation validation"""
    print_section("Example 4: Citation Validation")
    
    print("\nValidating that citations reference actual chunks:")
    
    # Good response with valid citations
    good_response = "Use vtkCylinderSource [1] as shown in the example [2]."
    
    # Bad response with invalid citations  
    bad_response = "Use vtkCylinderSource [1] and vtkConeSource [5]."  # [5] doesn't exist!
    
    if os.path.exists('.env'):
        generator = VTKRAGGenerator()
        
        metadata = {
            'query': 'test',
            'chunks_used': 2,
            'chunk_details': [
                {'position': 1, 'chunk_id': 'chunk1', 'source_type': 'api_doc', 'score': 8.0},
                {'position': 2, 'chunk_id': 'chunk2', 'source_type': 'example', 'score': 7.0}
            ]
        }
        
        # Validate good response
        good_resp = GeneratedResponse(
            answer=good_response,
            citations_found=[1, 2],
            citation_count=2,
            refused=False,
            metadata=metadata,
            raw_response=good_response,
            sources_used=""
        )
        
        validation_good = generator.validate_citations(good_resp, max_position=2)
        print(f"\nGood response: {good_response}")
        print(f"Validation: {validation_good['validation_passed']} ✓")
        
        # Validate bad response
        bad_resp = GeneratedResponse(
            answer=bad_response,
            citations_found=[1, 5],
            citation_count=2,
            refused=False,
            metadata=metadata,
            raw_response=bad_response,
            sources_used=""
        )
        
        validation_bad = generator.validate_citations(bad_resp, max_position=2)
        print(f"\nBad response: {bad_response}")
        print(f"Validation: {validation_bad['validation_passed']} ✗")
        print(f"Invalid citations: {validation_bad['invalid_citations']} (hallucinated!)")
    else:
        print("\n(Skipping - no .env file)")
        print("Citation [5] would be invalid (only 2 chunks available)")


def example5_complete_pipeline():
    """Example 5: Complete RAG pipeline integration"""
    print_section("Example 5: Complete RAG Pipeline Integration")
    
    print("\nComplete workflow from query to grounded response:")
    print("""
# Complete RAG Pipeline
import sys
sys.path.append('retrieval-pipeline')
sys.path.append('grounding-prompting')

from task_specific_retriever import TaskSpecificRetriever
from prompt_templates import VTKPromptTemplate, PromptConfig
from generator import VTKRAGGenerator

# Stage 1-3: Retrieve
retriever = TaskSpecificRetriever()
chunks = retriever.retrieve(
    query="How to create cylinder?",
    content_types=['CODE', 'API_DOC'],
    top_k=5
)

# Stage 4: Format context and generate prompt
context = "\\n\\n".join([
    f"[{i}] Source: {chunk.content_type} | {chunk.chunk_id}\\n{chunk.content}"
    for i, chunk in enumerate(chunks, 1)
])

template = VTKPromptTemplate(config=PromptConfig(
    require_citations=True,
    allow_speculation=False
))

prompt = template.generate_code_prompt(
    query="How to create cylinder?",
    context=context,
    style="complete"
)

# Create metadata
metadata = {
    'query': "How to create cylinder?",
    'chunks_used': len(chunks),
    'chunk_details': [
        {
            'position': i,
            'chunk_id': chunk.chunk_id,
            'source_type': chunk.content_type,
            'score': chunk.score
        }
        for i, chunk in enumerate(chunks, 1)
    ]
}

# Stage 5: Generate with LLM
generator = VTKRAGGenerator()
response = generator.generate(prompt, metadata)

# Display result
print(generator.format_response(response, include_sources=True))

# Validate citations
validation = generator.validate_citations(response, max_position=len(chunks))
if not validation['all_valid']:
    print("⚠️  Warning: Invalid citations detected!")
""")


def main():
    print("=" * 80)
    print("LLM Generation - Example Usage")
    print("=" * 80)
    
    # Check for .env
    if not os.path.exists('.env'):
        print("\n⚠️  No .env file found!")
        print("\nTo run with real LLM:")
        print("  1. cp .env.example .env")
        print("  2. Edit .env and add your API key")
        print("  3. Run this script again")
        print("\nRunning with mock examples (no API calls)...\n")
    
    try:
        example1_mock_generation()
        example2_refusal_response()
        example3_configuration_options()
        example4_citation_validation()
        example5_complete_pipeline()
        
        # Summary
        print_section("Summary")
        print("\n✅ Examples completed!")
        print("\nKey features:")
        print("  • Multi-provider LLM support (OpenAI, Anthropic, Local)")
        print("  • Forced citations with [N] notation")
        print("  • Refusal when context insufficient")
        print("  • Citation validation")
        print("  • Response formatting")
        print("  • Provenance tracking")
        
        print("\nNext steps:")
        print("  1. Create .env file with your API key")
        print("  2. Test with: python llm-generation/generator.py")
        print("  3. Integrate with full RAG pipeline")
        print("  4. Tune temperature and other parameters")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
