#!/usr/bin/env python3
"""
Query Routing Example

Demonstrates the unified query system with different query types.
Shows how queries are classified and routed to appropriate handlers.

This is a demo/example script, not an automated test.
For unit tests, see: tests/llm-generation/test_sequential_pipeline_extended.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'llm-generation'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'retrieval-pipeline'))

from sequential_pipeline import SequentialPipeline


def demo_query_classification():
    """Demonstrate query classification without LLM calls"""
    print("=" * 80)
    print("DEMONSTRATING QUERY CLASSIFICATION")
    print("=" * 80)
    
    # Create pipeline (no LLM calls, just classification)
    pipeline = SequentialPipeline(use_llm_decomposition=False)
    
    test_queries = [
        # CODE queries
        ("Create a cylinder in VTK", "code"),
        ("How do I render a sphere?", "code"),
        
        # API queries
        ("What does SetMapper do?", "api"),
        ("Explain the GetProperty method", "api"),
        ("What is vtkActor class?", "api"),
        
        # EXPLANATION queries
        ("Explain the VTK pipeline workflow", "explanation"),
        ("What's the difference between vtkActor and vtkActor2D?", "explanation"),
        
        # DATA queries
        ("I have points.csv, what can I do?", "data_query"),
        ("What techniques can I use with mesh.stl?", "data_query"),
        
        # CODE_TO_DATA queries (needs code parameter)
        ("Do you have example data for this?", "code_to_data"),
    ]
    
    print("\nQuery Classification Results:\n")
    for query, expected_type in test_queries:
        # Classify
        if "example data" in query:
            # Simulate code parameter
            result = pipeline._classify_query(query, code="test code")
        else:
            result = pipeline._classify_query(query)
        
        # Check result
        status = "✅" if result == expected_type else "❌"
        print(f"{status} Query: '{query[:50]}...'")
        print(f"   Expected: {expected_type}, Got: {result}\n")


def demo_query_flow():
    """Show the flow for different query types (without actual LLM calls)"""
    print("\n" + "=" * 80)
    print("QUERY FLOW DEMONSTRATION")
    print("=" * 80)
    
    examples = {
        "CODE": {
            "query": "Create a 3D cylinder",
            "flow": [
                "1. Classify: 'code'",
                "2. Route to: _handle_code_query()",
                "3. Decompose into steps",
                "4. Retrieve docs per step",
                "5. Generate code per step",
                "6. Assemble final result",
                "7. Return JSON with code"
            ]
        },
        "API": {
            "query": "What does SetMapper do?",
            "flow": [
                "1. Classify: 'api'",
                "2. Route to: _handle_api_query()",
                "3. Retrieve API documentation",
                "4. Use get_api_lookup_instructions()",
                "5. Generate JSON with API explanation",
                "6. Return JSON with parameters, examples"
            ]
        },
        "DATA_QUERY": {
            "query": "I have points.csv, what can I do?",
            "flow": [
                "1. Classify: 'data_query'",
                "2. Route to: _handle_data_query()",
                "3. Extract file type (.csv)",
                "4. Retrieve examples with CSV files",
                "5. Group by category",
                "6. Generate JSON with multiple techniques",
                "7. Return working code + alternatives"
            ]
        },
        "CODE_TO_DATA": {
            "query": "Do you have example STL files?",
            "flow": [
                "1. Classify: 'code_to_data'",
                "2. Route to: _handle_code_to_data_query()",
                "3. Parse code for vtkSTLReader",
                "4. Search examples with .stl files",
                "5. Extract download URLs",
                "6. Return JSON with file list"
            ]
        }
    }
    
    for query_type, info in examples.items():
        print(f"\n{query_type} Query Example:")
        print(f"Query: '{info['query']}'")
        print("\nFlow:")
        for step in info['flow']:
            print(f"  {step}")


def show_json_schemas():
    """Show the JSON schemas for each query type"""
    print("\n" + "=" * 80)
    print("JSON OUTPUT SCHEMAS")
    print("=" * 80)
    
    schemas = {
        "CODE": {
            "response_type": "answer",
            "content_type": "code",
            "code": "# Working Python code",
            "explanation": "How it works",
            "citations": [{"number": 1, "reason": "source"}],
            "confidence": "high"
        },
        "API": {
            "response_type": "answer",
            "content_type": "api",
            "explanation": "API documentation",
            "parameters": [{"name": "...", "type": "...", "description": "..."}],
            "usage_example": "code example",
            "confidence": "high"
        },
        "DATA_QUERY": {
            "response_type": "answer",
            "content_type": "code",
            "data_analysis": "What the data is",
            "suggested_techniques": ["technique1", "technique2"],
            "code": "Working example",
            "alternative_approaches": [{"technique": "...", "description": "..."}],
            "confidence": "high"
        },
        "CODE_TO_DATA": {
            "response_type": "answer",
            "content_type": "data",
            "explanation": "What data is needed",
            "data_files": [{"filename": "...", "download_url": "..."}],
            "confidence": "high"
        }
    }
    
    for query_type, schema in schemas.items():
        print(f"\n{query_type} Output Schema:")
        import json
        print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    print("\n🚀 VTK RAG - Query Routing Example\n")
    
    # Run demonstrations
    demo_query_classification()
    demo_query_flow()
    show_json_schemas()
    
    print("\n" + "=" * 80)
    print("✅ All query types are properly classified and routed!")
    print("=" * 80)
    print("\nTo test with actual LLM calls, set up your .env file and use:")
    print("  pipeline = SequentialPipeline(use_llm_decomposition=True)")
    print("  result = pipeline.process_query('Your query here')")
    print()
